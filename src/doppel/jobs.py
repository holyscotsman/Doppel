"""Long-running jobs. Each job records progress in the scans table and is
idempotent: re-running skips or upserts, never duplicates."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any

from doppel.drive import DriveClient


def _now() -> str:
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
        (_now(), stage),
    )
    cur = conn.execute(
        "INSERT INTO scans (stage, status, processed, started_at) "
        "VALUES (?, 'running', 0, ?)",
        (stage, _now()),
    )
    conn.commit()
    return int(cur.lastrowid)


def finish_scan(conn: sqlite3.Connection, scan_id: int, total: int) -> None:
    conn.execute(
        "UPDATE scans SET status = 'done', total = ?, finished_at = ? WHERE id = ?",
        (total, _now(), scan_id),
    )
    conn.commit()


def fail_scan(conn: sqlite3.Connection, scan_id: int, error: str) -> None:
    conn.execute(
        "UPDATE scans SET status = 'failed', error = ?, finished_at = ? WHERE id = ?",
        (error, _now(), scan_id),
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


def run_sync(conn: sqlite3.Connection, client: DriveClient) -> int:
    """Full inventory sync: upsert every Drive image, mark unseen rows missing.

    Returns the scans row id. Idempotent: a re-run with an unchanged Drive
    account changes nothing.
    """
    scan_id = start_scan(conn, "sync")
    try:
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS seen (drive_id TEXT PRIMARY KEY)")
        conn.execute("DELETE FROM seen")
        processed = 0
        page_token: str | None = None
        while True:
            page = client.list_images_page(page_token)
            for file in page.get("files", []):
                _upsert_photo(conn, file)
                conn.execute(
                    "INSERT OR IGNORE INTO seen (drive_id) VALUES (?)", (file["id"],)
                )
                processed += 1
            conn.execute(
                "UPDATE scans SET processed = ? WHERE id = ?", (processed, scan_id)
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
