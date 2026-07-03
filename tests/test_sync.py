import sqlite3

import pytest

from doppel.jobs import run_sync
from tests.fakes import FakeDriveClient, make_file


def photos(conn: sqlite3.Connection) -> dict[str, sqlite3.Row]:
    return {row["drive_id"]: row for row in conn.execute("SELECT * FROM photos")}


def test_sync_paginates_to_completion(conn) -> None:
    files = [make_file(f"id{i}", name=f"p{i}.jpg") for i in range(5)]
    client = FakeDriveClient(files, page_size=2)

    run_sync(conn, client)

    assert len(photos(conn)) == 5
    assert client.pages_served == [None, "2", "4"]


def test_sync_stores_metadata(conn) -> None:
    client = FakeDriveClient(
        [
            make_file(
                "a",
                name="x.png",
                mime_type="image/png",
                size=42,
                md5="abc",
                width=100,
                height=50,
                thumbnail_link="https://t/x=s220",
            )
        ]
    )

    run_sync(conn, client)

    row = photos(conn)["a"]
    assert row["name"] == "x.png"
    assert row["mime_type"] == "image/png"
    assert row["size"] == 42
    assert row["md5"] == "abc"
    assert row["width"] == 100
    assert row["height"] == 50
    assert row["thumbnail_link"] == "https://t/x=s220"
    assert row["status"] == "active"


def test_sync_is_idempotent(conn) -> None:
    files = [make_file(f"id{i}") for i in range(3)]
    run_sync(conn, FakeDriveClient(files))
    before = {k: tuple(v) for k, v in photos(conn).items()}

    run_sync(conn, FakeDriveClient(files))

    after = {k: tuple(v) for k, v in photos(conn).items()}
    assert before == after


def test_sync_updates_changed_metadata(conn) -> None:
    run_sync(conn, FakeDriveClient([make_file("a", name="old.jpg")]))
    run_sync(conn, FakeDriveClient([make_file("a", name="new.jpg")]))

    assert photos(conn)["a"]["name"] == "new.jpg"
    assert len(photos(conn)) == 1


def test_sync_marks_unseen_as_missing_and_revives(conn) -> None:
    run_sync(conn, FakeDriveClient([make_file("a"), make_file("b")]))
    run_sync(conn, FakeDriveClient([make_file("a")]))

    assert photos(conn)["a"]["status"] == "active"
    assert photos(conn)["b"]["status"] == "missing"

    run_sync(conn, FakeDriveClient([make_file("a"), make_file("b")]))
    assert photos(conn)["b"]["status"] == "active"


def test_sync_records_progress(conn) -> None:
    files = [make_file(f"id{i}") for i in range(5)]
    scan_id = run_sync(conn, FakeDriveClient(files, page_size=2))

    scan = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
    assert scan["stage"] == "sync"
    assert scan["status"] == "done"
    assert scan["processed"] == 5
    assert scan["total"] == 5
    assert scan["started_at"] and scan["finished_at"]


def test_sync_failure_is_recorded_and_recoverable(conn) -> None:
    files = [make_file(f"id{i}") for i in range(4)]
    failing = FakeDriveClient(files, page_size=2)
    failing.fail_after_pages = 1

    with pytest.raises(RuntimeError):
        run_sync(conn, failing)

    scan = conn.execute("SELECT * FROM scans ORDER BY id DESC LIMIT 1").fetchone()
    assert scan["status"] == "failed"
    assert "simulated" in scan["error"]
    # photos from the completed page survive; a re-run recovers cleanly
    assert len(photos(conn)) == 2
    run_sync(conn, FakeDriveClient(files, page_size=2))
    assert len(photos(conn)) == 4
    assert all(row["status"] == "active" for row in photos(conn).values())


def test_sync_handles_files_without_optional_fields(conn) -> None:
    client = FakeDriveClient(
        [make_file("bare", size=None, md5=None, width=None, height=None)]
    )
    run_sync(conn, client)

    row = photos(conn)["bare"]
    assert row["size"] is None
    assert row["md5"] is None
    assert row["status"] == "active"


def test_interrupt_marks_scan_failed(conn) -> None:
    class InterruptingClient:
        def list_images_page(self, page_token=None, parent_ids=None):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_sync(conn, InterruptingClient())

    scan = conn.execute("SELECT * FROM scans ORDER BY id DESC LIMIT 1").fetchone()
    assert scan["status"] == "failed"
    assert "KeyboardInterrupt" in scan["error"]


def test_orphaned_running_scan_is_reconciled_on_next_run(conn) -> None:
    # simulate a SIGKILLed process: a committed 'running' row nothing updated
    conn.execute(
        "INSERT INTO scans (stage, status, processed, started_at) "
        "VALUES ('sync', 'running', 3, '2024-01-01T00:00:00+00:00')"
    )
    conn.commit()

    run_sync(conn, FakeDriveClient([make_file("a")]))

    rows = conn.execute("SELECT status, error FROM scans ORDER BY id").fetchall()
    assert rows[0]["status"] == "failed"
    assert rows[0]["error"] == "interrupted"
    assert rows[1]["status"] == "done"


def test_failed_partial_sync_leaves_prior_statuses_untouched(conn) -> None:
    files = [make_file(f"id{i}") for i in range(4)]
    run_sync(conn, FakeDriveClient(files, page_size=2))

    failing = FakeDriveClient(files, page_size=2)
    failing.fail_after_pages = 1
    with pytest.raises(RuntimeError):
        run_sync(conn, failing)

    # photos on pages never fetched must not be marked missing by a failed run
    assert all(row["status"] == "active" for row in photos(conn).values())
    assert len(photos(conn)) == 4
