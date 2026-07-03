"""CLI entry point for `make scan`: run a full Drive inventory sync."""

from __future__ import annotations

import sys

from doppel.config import load_config
from doppel.db import connect
from doppel.drive import GoogleDriveClient, get_credentials
from doppel.jobs import run_sync


def main() -> int:
    config = load_config()
    conn = connect(config.db_path)
    try:
        creds = get_credentials()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    client = GoogleDriveClient(creds)
    scan_id = run_sync(conn, client)
    row = conn.execute(
        "SELECT processed FROM scans WHERE id = ?", (scan_id,)
    ).fetchone()
    active = conn.execute(
        "SELECT COUNT(*) AS n FROM photos WHERE status = 'active'"
    ).fetchone()
    missing = conn.execute(
        "SELECT COUNT(*) AS n FROM photos WHERE status = 'missing'"
    ).fetchone()
    print(
        f"sync complete: {row['processed']} listed, "
        f"{active['n']} active, {missing['n']} missing"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
