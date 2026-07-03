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
    conn.execute(
        """
        INSERT INTO photos (drive_id, name, mime_type, size, md5, width, height,
                            created_time, modified_time, thumbnail_link, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
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
