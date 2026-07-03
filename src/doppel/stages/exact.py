"""Stage 1 — exact duplicates. Pure SQL over Drive md5 metadata; no image
bytes required. md5 identifies byte-identical files only."""

from __future__ import annotations

import sqlite3

from doppel.jobs import fail_scan, finish_scan, now, start_scan


def rebuild_groups(conn: sqlite3.Connection, tier: str) -> None:
    """Delete a tier's groups and members. Decisions are keyed by photo and
    deliberately survive rebuilds."""
    conn.execute(
        "DELETE FROM group_members WHERE group_id IN "
        "(SELECT id FROM groups WHERE tier = ?)",
        (tier,),
    )
    conn.execute("DELETE FROM groups WHERE tier = ?", (tier,))


def run_exact(conn: sqlite3.Connection) -> int:
    """Group active photos byte-identical by md5. Returns the scans row id."""
    scan_id = start_scan(conn, "exact")
    try:
        dupes = conn.execute(
            """
            SELECT md5, COUNT(*) AS n FROM photos
            WHERE md5 IS NOT NULL AND status = 'active'
            GROUP BY md5 HAVING n > 1
            ORDER BY md5
            """
        ).fetchall()
        # commit total before the rebuild transaction so the polling UI
        # (on its own connection) can see it; the rebuild itself stays
        # atomic in the transaction below
        conn.execute("UPDATE scans SET total = ? WHERE id = ?", (len(dupes), scan_id))
        conn.commit()
        rebuild_groups(conn, "exact")
        for row in dupes:
            cur = conn.execute(
                "INSERT INTO groups (tier, created_at) VALUES ('exact', ?)",
                (now(),),
            )
            group_id = cur.lastrowid
            conn.execute(
                """
                INSERT INTO group_members (group_id, photo_id)
                SELECT ?, id FROM photos
                WHERE md5 = ? AND status = 'active'
                """,
                (group_id, row["md5"]),
            )
        conn.execute(
            "UPDATE scans SET processed = ? WHERE id = ?", (len(dupes), scan_id)
        )
        conn.commit()
        finish_scan(conn, scan_id, total=len(dupes))
    except BaseException as exc:
        conn.rollback()
        fail_scan(conn, scan_id, f"{type(exc).__name__}: {exc}")
        raise
    return scan_id
