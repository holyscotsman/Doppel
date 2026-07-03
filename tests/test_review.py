"""Phase 5: decision persistence, preselect, filtering, CSV export."""

import csv
import io

import pytest
from fastapi.testclient import TestClient

from doppel.app import create_app
from doppel.db import connect
from tests.fakes import FakeImageFetcher, insert_photo


@pytest.fixture
def client(config):
    app = create_app(
        config=config, fetcher_factory=lambda cfg: FakeImageFetcher(cfg.cache_dir)
    )
    with TestClient(app) as test_client:
        test_client.app = app
        yield test_client


@pytest.fixture
def group(client, config):
    """An exact group of two photos; returns (group_id, big_id, small_id)."""
    conn = connect(config.db_path)
    big = insert_photo(conn, "big", name="big.jpg", md5="dup", size=5000)
    small = insert_photo(conn, "small", name="small.jpg", md5="dup", size=100)
    conn.close()
    client.post("/scans/exact")
    client.app.state.runner.wait(timeout=10)
    conn = connect(config.db_path)
    group_id = conn.execute("SELECT id FROM groups").fetchone()["id"]
    conn.close()
    return group_id, big, small


def decisions(config) -> dict[int, str]:
    conn = connect(config.db_path)
    rows = {
        row["photo_id"]: row["action"]
        for row in conn.execute("SELECT photo_id, action FROM decisions")
    }
    conn.close()
    return rows


def test_default_preselect_keeps_largest(client, group) -> None:
    group_id, big, small = group

    page = client.get(f"/groups/{group_id}")

    assert f'name="action_{big}" value="keep" checked' in page.text
    assert f'name="action_{small}" value="trash" checked' in page.text


def test_decisions_persist_and_render(client, config, group) -> None:
    group_id, big, small = group

    resp = client.post(
        f"/groups/{group_id}/decisions",
        data={f"action_{big}": "trash", f"action_{small}": "keep"},
        follow_redirects=False,
    )

    assert resp.status_code == 303
    assert decisions(config) == {big: "trash", small: "keep"}

    page = client.get(f"/groups/{group_id}")
    assert f'name="action_{big}" value="trash" checked' in page.text
    assert f'name="action_{small}" value="keep" checked' in page.text


def test_decisions_update_on_resubmit(client, config, group) -> None:
    group_id, big, small = group
    client.post(f"/groups/{group_id}/decisions", data={f"action_{big}": "trash"})
    client.post(f"/groups/{group_id}/decisions", data={f"action_{big}": "keep"})

    assert decisions(config)[big] == "keep"


def test_invalid_action_rejected(client, group) -> None:
    group_id, big, _ = group

    resp = client.post(f"/groups/{group_id}/decisions", data={f"action_{big}": "shred"})

    assert resp.status_code == 422


def test_foreign_photo_field_ignored(client, config, group) -> None:
    group_id, big, _ = group
    conn = connect(config.db_path)
    outsider = insert_photo(conn, "outsider", md5="zzz")
    conn.close()

    client.post(
        f"/groups/{group_id}/decisions",
        data={f"action_{big}": "keep", f"action_{outsider}": "trash"},
    )

    assert outsider not in decisions(config)


def test_reviewed_filter(client, config, group) -> None:
    group_id, big, small = group
    conn = connect(config.db_path)
    insert_photo(conn, "c1", md5="dup2", size=10)
    insert_photo(conn, "c2", md5="dup2", size=20)
    conn.close()
    client.post("/scans/exact")
    client.app.state.runner.wait(timeout=10)
    # group ids may have been rebuilt: rediscover the group containing 'big'
    conn = connect(config.db_path)
    group_id = conn.execute(
        """
        SELECT m.group_id FROM group_members m
        JOIN photos p ON p.id = m.photo_id WHERE p.drive_id = 'big'
        """
    ).fetchone()["group_id"]
    conn.close()
    client.post(
        f"/groups/{group_id}/decisions",
        data={f"action_{big}": "keep", f"action_{small}": "trash"},
    )

    unreviewed = client.get("/groups", params={"tier": "exact", "reviewed": "no"})
    reviewed = client.get("/groups", params={"tier": "exact", "reviewed": "yes"})
    everything = client.get("/groups", params={"tier": "exact"})

    assert "(1)" in unreviewed.text
    assert "(1)" in reviewed.text
    assert "(2)" in everything.text
    assert f"group {group_id}" in reviewed.text
    assert f"group {group_id}" not in unreviewed.text


def test_export_csv_lists_only_trash(client, config, group) -> None:
    group_id, big, small = group
    client.post(
        f"/groups/{group_id}/decisions",
        data={f"action_{big}": "keep", f"action_{small}": "trash"},
    )

    resp = client.get("/export")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    rows = list(csv.reader(io.StringIO(resp.text)))
    assert rows[0] == ["drive_id", "name", "size", "md5", "url"]
    assert len(rows) == 2
    assert rows[1] == [
        "small",
        "small.jpg",
        "100",
        "dup",
        "https://drive.google.com/file/d/small/view",
    ]


def test_full_flow_scan_review_export(client, config) -> None:
    """SPEC Phase 5 acceptance: scan -> review groups -> export CSV."""
    from doppel.db import connect as db_connect
    from doppel.jobs import run_sync
    from tests.fakes import FakeDriveClient, make_file

    conn = db_connect(config.db_path)
    run_sync(
        conn,
        FakeDriveClient(
            [
                make_file("d1", name="a.jpg", md5="x", size=900),
                make_file("d2", name="a copy.jpg", md5="x", size=900),
                make_file("d3", name="unrelated.jpg", md5="y", size=50),
            ]
        ),
    )
    conn.close()

    client.post("/scans/exact")
    client.app.state.runner.wait(timeout=10)

    conn = db_connect(config.db_path)
    group_id = conn.execute("SELECT id FROM groups").fetchone()["id"]
    ids = {
        row["drive_id"]: row["id"]
        for row in conn.execute("SELECT id, drive_id FROM photos")
    }
    conn.close()

    detail = client.get(f"/groups/{group_id}")
    assert "a.jpg" in detail.text
    client.post(
        f"/groups/{group_id}/decisions",
        data={f"action_{ids['d1']}": "keep", f"action_{ids['d2']}": "trash"},
    )

    export = client.get("/export").text
    assert "a copy.jpg" in export
    assert "unrelated.jpg" not in export

    dashboard = client.get("/")
    assert "marked for trash" in dashboard.text
