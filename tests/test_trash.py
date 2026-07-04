"""Move-to-Trash: confirm page, execution, partial failures, and the
guarantee that the app only ever trashes (reversible), never hard-deletes."""

import pytest
from fastapi.testclient import TestClient

from doppel.app import create_app
from doppel.db import connect
from doppel.jobs import now
from tests.fakes import FakeImageFetcher, FakeTrashClient, insert_photo


def mark_trash(conn, drive_id, size, name=None, folder_path=None, status="active"):
    """Insert a photo and record a 'trash' decision for it; return its id."""
    pid = insert_photo(
        conn,
        drive_id,
        name=name or f"{drive_id}.jpg",
        size=size,
        status=status,
        folder_path=folder_path,
    )
    conn.execute(
        "INSERT INTO decisions (photo_id, action, decided_at) VALUES (?, 'trash', ?)",
        (pid, now()),
    )
    conn.commit()
    return pid


@pytest.fixture
def trash_client(config):
    """App wired with a recording trash client, so no real Drive write happens."""
    fake = FakeTrashClient()
    app = create_app(
        config=config,
        fetcher_factory=lambda cfg: FakeImageFetcher(cfg.cache_dir),
        trash_client_factory=lambda: fake,
    )
    with TestClient(app) as c:
        c.fake_trash = fake
        yield c


def test_confirm_lists_pending_trash(trash_client, config):
    conn = connect(config.db_path)
    mark_trash(
        conn, "big", 5_000_000, name="big.jpg", folder_path="Photos / 2024 / Beach"
    )
    mark_trash(conn, "small", 1_000_000, name="small.jpg")
    conn.close()

    page = trash_client.get("/trash/confirm")
    assert page.status_code == 200
    assert "big.jpg" in page.text and "small.jpg" in page.text
    assert "photos marked to trash" in page.text
    assert ">2</b> photos marked to trash" in page.text
    assert "Photos / 2024 / Beach" in page.text
    # reassurance that it's reversible is front and centre
    assert "reversible" in page.text.lower()
    assert "30 days" in page.text


def test_confirm_empty_state(trash_client, config):
    page = trash_client.get("/trash/confirm")
    assert page.status_code == 200
    assert "Nothing is marked for trash" in page.text


def test_trash_moves_files_and_marks_status(trash_client, config):
    conn = connect(config.db_path)
    a = mark_trash(conn, "aaa", 3000)
    b = mark_trash(conn, "bbb", 2000)
    conn.close()

    resp = trash_client.post("/trash")
    assert resp.status_code == 200
    # both were moved to Trash via the client (reversible), in size order
    assert trash_client.fake_trash.trashed == ["aaa", "bbb"]
    assert "2 photos moved to Trash" in resp.text

    conn = connect(config.db_path)
    statuses = {
        r["id"]: r["status"]
        for r in conn.execute(
            "SELECT id, status FROM photos WHERE id IN (?, ?)", (a, b)
        )
    }
    conn.close()
    assert statuses[a] == "trashed" and statuses[b] == "trashed"


def test_trash_never_hard_deletes(trash_client, config):
    """The trash client exposes only trash_file — there is no delete path. The
    route must never reach for one."""
    conn = connect(config.db_path)
    mark_trash(conn, "xyz", 1000)
    conn.close()

    trash_client.post("/trash")
    # only trash_file was ever called; the fake would have no delete method
    assert trash_client.fake_trash.trashed == ["xyz"]
    assert not hasattr(trash_client.fake_trash, "delete_file")


def test_trash_partial_failure_keeps_failed_active(config):
    fake = FakeTrashClient(fail_ids={"locked"})
    app = create_app(
        config=config,
        fetcher_factory=lambda cfg: FakeImageFetcher(cfg.cache_dir),
        trash_client_factory=lambda: fake,
    )
    conn = connect(config.db_path)
    ok = mark_trash(conn, "ok", 2000)
    locked = mark_trash(conn, "locked", 9000)  # largest, tried first, fails
    conn.close()

    with TestClient(app) as c:
        resp = c.post("/trash")
    assert resp.status_code == 200
    assert "1 photo moved to Trash" in resp.text
    assert "couldn't be moved" in resp.text

    conn = connect(config.db_path)
    statuses = {
        r["id"]: r["status"]
        for r in conn.execute(
            "SELECT id, status FROM photos WHERE id IN (?, ?)", (ok, locked)
        )
    }
    conn.close()
    assert statuses[ok] == "trashed"  # succeeded
    assert statuses[locked] == "active"  # failed one is untouched, still reviewable


def test_trash_post_with_nothing_pending_redirects(trash_client):
    resp = trash_client.post("/trash", follow_redirects=False)
    assert resp.status_code == 303
    assert trash_client.fake_trash.trashed == []


def test_trashed_photos_drop_out_of_dashboard_count(trash_client, config):
    conn = connect(config.db_path)
    mark_trash(conn, "one", 1000)
    mark_trash(conn, "two", 1000)
    conn.close()

    assert "marked for trash" in trash_client.get("/").text
    trash_client.post("/trash")
    # after moving, the pending-trash count is 0 (they're status='trashed')
    conn = connect(config.db_path)
    pending = conn.execute(
        "SELECT COUNT(*) AS n FROM decisions d JOIN photos p ON p.id = d.photo_id "
        "WHERE d.action = 'trash' AND p.status = 'active'"
    ).fetchone()["n"]
    conn.close()
    assert pending == 0


def test_can_trash_false_without_connection(config, tmp_path, monkeypatch):
    from doppel.app import can_trash

    monkeypatch.chdir(tmp_path)  # no token.json / service_account.json
    assert can_trash() is False
