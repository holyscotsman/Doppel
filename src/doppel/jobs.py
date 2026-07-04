"""Long-running jobs. Each job records progress in the scans table and is
idempotent: re-running skips or upserts, never duplicates."""

from __future__ import annotations

import sqlite3
import threading
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from doppel.drive import PARENT_QUERY_CHUNK, DriveClient, collect_folder_tree


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def start_scan(conn: sqlite3.Connection, stage: str) -> int:
    """Insert a running scans row and return its id.

    Any prior 'running' row for the same stage is an orphan from a killed
    process (jobs run one at a time); mark it failed so the ledger stays
    truthful.
    """
    conn.execute(
        "UPDATE scans SET status = 'failed', error = 'interrupted', finished_at = ? "
        "WHERE stage = ? AND status = 'running'",
        (now(), stage),
    )
    cur = conn.execute(
        "INSERT INTO scans (stage, status, processed, started_at) "
        "VALUES (?, 'running', 0, ?)",
        (stage, now()),
    )
    conn.commit()
    return int(cur.lastrowid)


def finish_scan(conn: sqlite3.Connection, scan_id: int, total: int) -> None:
    conn.execute(
        "UPDATE scans SET status = 'done', total = ?, finished_at = ? WHERE id = ?",
        (total, now(), scan_id),
    )
    conn.commit()


def fail_scan(conn: sqlite3.Connection, scan_id: int, error: str) -> None:
    conn.execute(
        "UPDATE scans SET status = 'failed', error = ?, finished_at = ? WHERE id = ?",
        (error, now(), scan_id),
    )
    conn.commit()


def reconcile_orphaned_scans(conn: sqlite3.Connection) -> None:
    """Mark every 'running' scans row failed. Call at process startup, when
    no job can actually be running — a killed process leaves its row behind
    and the dashboard would report a phantom in-progress scan forever."""
    conn.execute(
        "UPDATE scans SET status = 'failed', error = 'interrupted', "
        "finished_at = ? WHERE status = 'running'",
        (now(),),
    )
    conn.commit()


def _upsert_photo(conn: sqlite3.Connection, file: dict[str, Any]) -> None:
    meta = file.get("imageMediaMetadata") or {}
    size = file.get("size")
    parents = file.get("parents") or []
    parent_id = parents[0] if parents else None
    conn.execute(
        """
        INSERT INTO photos (drive_id, name, mime_type, size, md5, width, height,
                            created_time, modified_time, thumbnail_link,
                            parent_id, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
        ON CONFLICT(drive_id) DO UPDATE SET
          name = excluded.name,
          mime_type = excluded.mime_type,
          size = excluded.size,
          md5 = excluded.md5,
          width = excluded.width,
          height = excluded.height,
          created_time = excluded.created_time,
          modified_time = excluded.modified_time,
          thumbnail_link = excluded.thumbnail_link,
          parent_id = excluded.parent_id,
          status = 'active'
        """,
        (
            file["id"],
            file["name"],
            file["mimeType"],
            int(size) if size is not None else None,
            file.get("md5Checksum"),
            meta.get("width"),
            meta.get("height"),
            file.get("createdTime"),
            file.get("modifiedTime"),
            file.get("thumbnailLink"),
            parent_id,
        ),
    )


def _invalidate_derived(
    conn: sqlite3.Connection, drive_id: str, cache_dir: Path | str | None
) -> None:
    """A file was edited in place (same drive_id, new md5): every derived
    artifact describes the OLD pixels. Drop hashes, embedding, and cached
    thumbnails so the next stage runs recompute them."""
    conn.execute(
        "UPDATE photos SET phash = NULL, dhash = NULL, thumb_path = NULL "
        "WHERE drive_id = ?",
        (drive_id,),
    )
    row = conn.execute(
        "SELECT id FROM photos WHERE drive_id = ?", (drive_id,)
    ).fetchone()
    if row is not None:
        from doppel.db import ensure_vec_schema

        ensure_vec_schema(conn)
        conn.execute("DELETE FROM embeddings WHERE photo_id = ?", (row["id"],))
    if cache_dir is not None:
        for stale in Path(cache_dir).glob(f"{drive_id}_*"):
            stale.unlink(missing_ok=True)


def stage_was_interrupted(conn: sqlite3.Connection, stage: str) -> bool:
    """True only when this stage's most recent scan was abnormally terminated —
    a killed/crashed process, whose 'running' row is reconciled to
    error='interrupted' (start_scan and reconcile_orphaned_scans write exactly
    that sentinel). A run that failed for a REAL reason (a Drive or VLM error)
    carries a different error string and is NOT treated as a resume — it should
    just be retried, not have its recent work thrown away and redone."""
    row = conn.execute(
        "SELECT status, error FROM scans WHERE stage = ? ORDER BY id DESC LIMIT 1",
        (stage,),
    ).fetchone()
    return (
        row is not None and row["status"] == "failed" and row["error"] == "interrupted"
    )


def _drop_cached_thumbs(
    conn: sqlite3.Connection, ids: list[int], cache_dir: Path | str | None
) -> None:
    if not ids or cache_dir is None:
        return
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT drive_id FROM photos WHERE id IN ({placeholders})",  # noqa: S608
        ids,
    ).fetchall()
    for row in rows:
        for stale in Path(cache_dir).glob(f"{row['drive_id']}_*"):
            stale.unlink(missing_ok=True)


def reprocess_tail(
    conn: sqlite3.Connection,
    stage: str,
    n: int,
    cache_dir: Path | str | None = None,
) -> list[int]:
    """Invalidate the last `n` finished photos of a resumed stage so they are
    recomputed from a fresh fetch — cheap insurance against a partial write or
    truncated *download* at the interruption boundary. Returns the invalidated
    photo ids.

    near    -> clear phash/dhash + delete cached thumbnail of the highest-id
               hashed photos (the cache delete is what forces a re-fetch)
    similar -> delete the highest-id stored embeddings + cached thumbnails
    Other stages no-op. In particular adjudicate is intentionally excluded:
    a VLM verdict is schema-validated JSON, not a downloaded byte stream, so it
    has no truncation-corruption failure mode — and deleting a verdict whose
    pair is no longer a candidate would silently lose it. Adjudicate resumes
    safely by simply re-asking pairs that lack a verdict.

    Call once per stage run, immediately before the re-derive step: it selects
    "highest id that still has output", so calling it twice would walk further
    back each time. ("Highest id", not literally "last to finish" — near/similar
    write in completion order, but a torn low-id write is impossible under
    SQLite atomicity and is caught by the phash IS NULL / NOT IN embeddings
    resume query anyway.)
    """
    if n <= 0:
        return []
    if stage == "near":
        ids = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM photos WHERE status = 'active' AND phash IS NOT NULL "
                "ORDER BY id DESC LIMIT ?",
                (n,),
            )
        ]
        for pid in ids:
            conn.execute(
                "UPDATE photos SET phash = NULL, dhash = NULL, thumb_path = NULL "
                "WHERE id = ?",
                (pid,),
            )
        _drop_cached_thumbs(conn, ids, cache_dir)
        conn.commit()
        return ids
    if stage == "similar":
        from doppel.db import ensure_vec_schema

        ensure_vec_schema(conn)
        ids = [
            r["id"]
            for r in conn.execute(
                "SELECT p.id FROM photos p WHERE p.status = 'active' "
                "AND p.id IN (SELECT photo_id FROM embeddings) "
                "ORDER BY p.id DESC LIMIT ?",
                (n,),
            )
        ]
        for pid in ids:
            conn.execute("DELETE FROM embeddings WHERE photo_id = ?", (pid,))
        _drop_cached_thumbs(conn, ids, cache_dir)
        conn.commit()
        return ids
    return []


def _resolve_folder_path(
    client: DriveClient,
    cache: dict[str, tuple[str | None, str | None]],
    folder_id: str | None,
    depth: int = 3,
) -> str | None:
    """Best-effort 'grandparent / parent / folder' label for a folder id.

    Walks up to `depth` levels via client.get_folder, memoising each folder's
    (name, parent) in `cache` so a shared ancestor is fetched once per sync.
    Returns None if nothing resolved (e.g. an ancestor is not readable by this
    account) — path display is cosmetic and must never fail a sync.
    """
    segments: list[str] = []
    fid = folder_id
    for _ in range(depth):
        if fid is None:
            break
        if fid not in cache:
            try:
                meta = client.get_folder(fid)
            except Exception:
                break
            parents = meta.get("parents") or []
            cache[fid] = (meta.get("name"), parents[0] if parents else None)
        name, parent = cache[fid]
        if not name:
            break
        segments.append(name)
        fid = parent
    if not segments:
        return None
    return " / ".join(reversed(segments))


def _resolve_folder_paths(conn: sqlite3.Connection, client: DriveClient) -> None:
    """Fill photos.folder_path for every active photo, grouped by parent so
    each distinct folder resolves once. Skipped if the client can't resolve
    folders (e.g. the test fake) — best-effort, never raises."""
    if not hasattr(client, "get_folder"):
        return
    # Path display is purely cosmetic: it must NEVER fail the inventory sync.
    # The whole body (not just the per-folder fetch) is guarded — a query error
    # here, e.g. write contention with a concurrent request, would otherwise
    # bubble up to fail_scan and mark a good sync failed.
    try:
        cache: dict[str, tuple[str | None, str | None]] = {}
        parents = [
            row["parent_id"]
            for row in conn.execute(
                "SELECT DISTINCT parent_id FROM photos "
                "WHERE status = 'active' AND parent_id IS NOT NULL"
            )
        ]
        for parent_id in parents:
            path = _resolve_folder_path(client, cache, parent_id)
            if path is not None:
                conn.execute(
                    "UPDATE photos SET folder_path = ? WHERE parent_id = ?",
                    (path, parent_id),
                )
        conn.commit()
    except Exception:
        return


def run_sync(
    conn: sqlite3.Connection,
    client: DriveClient,
    cache_dir: Path | str | None = None,
    folder_ids: list[str] | None = None,
) -> int:
    """Full inventory sync: upsert every Drive image, mark unseen rows missing.

    Returns the scans row id. Idempotent: a re-run with an unchanged Drive
    account changes nothing. When a file's content changed in place (same
    drive_id, different md5), its derived state is invalidated.

    folder_ids=None scans the whole Drive (one unscoped query). A list of
    folder ids scopes the scan to the union of those folders' subtrees
    (walked recursively — Drive has no recursive query). This is how
    service-account mode scans the folders shared with it, and how the user
    scopes to one folder. Photos synced earlier but outside the scope are
    marked 'missing': out of scope means out of the review set.
    """
    scan_id = start_scan(conn, "sync")
    try:
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS seen (drive_id TEXT PRIMARY KEY)")
        conn.execute("DELETE FROM seen")

        # each batch is one files.list query: the whole Drive (None), or a
        # chunk of parent-folder clauses from the scoped subtrees
        if folder_ids is None:
            batches: list[list[str] | None] = [None]
        else:
            all_folders: set[str] = set()
            for root in folder_ids:
                all_folders.update(collect_folder_tree(client, root))
            ordered = sorted(all_folders)
            batches = [
                ordered[i : i + PARENT_QUERY_CHUNK]
                for i in range(0, len(ordered), PARENT_QUERY_CHUNK)
            ]

        # an empty scope (folder_ids == []) means "no accessible folders" —
        # scan nothing and, crucially, DON'T mark the whole inventory missing.
        # (never pass parent_ids=[] to the client: the real client treats a
        # falsy parent list as an unscoped whole-Drive query.)
        if not batches:
            finish_scan(conn, scan_id, total=0)
            return scan_id

        processed = 0
        for parent_ids in batches:
            page_token: str | None = None
            while True:
                page = client.list_images_page(page_token, parent_ids=parent_ids)
                for file in page.get("files", []):
                    old = conn.execute(
                        "SELECT md5 FROM photos WHERE drive_id = ?", (file["id"],)
                    ).fetchone()
                    _upsert_photo(conn, file)
                    if old is not None and old["md5"] != file.get("md5Checksum"):
                        _invalidate_derived(conn, file["id"], cache_dir)
                    conn.execute(
                        "INSERT OR IGNORE INTO seen (drive_id) VALUES (?)",
                        (file["id"],),
                    )
                    processed += 1
                conn.execute(
                    "UPDATE scans SET processed = ? WHERE id = ?",
                    (processed, scan_id),
                )
                conn.commit()
                page_token = page.get("nextPageToken")
                if not page_token:
                    break
        conn.execute(
            "UPDATE photos SET status = 'missing' "
            "WHERE drive_id NOT IN (SELECT drive_id FROM seen)"
        )
        _resolve_folder_paths(conn, client)
        finish_scan(conn, scan_id, total=processed)
    except BaseException as exc:  # incl. KeyboardInterrupt: ledger must be truthful
        fail_scan(conn, scan_id, f"{type(exc).__name__}: {exc}")
        raise
    return scan_id


class JobRunner:
    """One background worker thread. Long-running stages execute here and
    report progress to the scans table; the UI polls that table."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stage: str | None = None

    def running_stage(self) -> str | None:
        """The stage currently executing, or None when idle."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self._stage
            return None

    def start(self, stage: str, target: Callable[[], None]) -> bool:
        """Run target on the worker thread. Returns False if a job is
        already running (one job at a time, per SPEC architecture)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False

            def _run() -> None:
                try:
                    target()
                except Exception:
                    # the job already recorded failure in the scans table;
                    # keep the worker thread from dying loudly
                    traceback.print_exc()

            self._stage = stage
            self._thread = threading.Thread(target=_run, daemon=True)
            self._thread.start()
            return True

    def wait(self, timeout: float | None = None) -> None:
        """Block until the current job finishes. Used by tests."""
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)
