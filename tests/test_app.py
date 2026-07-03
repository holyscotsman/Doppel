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


def seed_duplicates(config) -> None:
    conn = connect(config.db_path)
    insert_photo(conn, "a1", name="beach.jpg", md5="aaa", size=2000)
    insert_photo(conn, "a2", name="beach copy.jpg", md5="aaa", size=1000)
    insert_photo(conn, "solo", md5="zzz")
    conn.close()


def run_exact_via_ui(client) -> None:
    resp = client.post("/scans/exact")
    assert resp.status_code == 200
    client.app.state.runner.wait(timeout=10)


def test_dashboard_renders_counts(client, config) -> None:
    seed_duplicates(config)
    run_exact_via_ui(client)

    resp = client.get("/")
    assert resp.status_code == 200
    assert "3" in resp.text  # active photos
    assert "exact groups" in resp.text


def test_group_list_and_detail(client, config) -> None:
    seed_duplicates(config)
    run_exact_via_ui(client)

    listing = client.get("/groups", params={"tier": "exact"})
    assert listing.status_code == 200
    assert "2 photos" in listing.text

    conn = connect(config.db_path)
    group_id = conn.execute("SELECT id FROM groups").fetchone()["id"]
    conn.close()

    detail = client.get(f"/groups/{group_id}")
    assert detail.status_code == 200
    assert "beach.jpg" in detail.text
    assert "beach copy.jpg" in detail.text


def test_thumb_serves_image_bytes(client, config) -> None:
    seed_duplicates(config)
    conn = connect(config.db_path)
    photo_id = conn.execute("SELECT id FROM photos LIMIT 1").fetchone()["id"]
    conn.close()

    resp = client.get(f"/thumb/{photo_id}")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content[:2] == b"\xff\xd8"  # JPEG magic


def test_scan_status_endpoint(client, config) -> None:
    seed_duplicates(config)
    run_exact_via_ui(client)

    conn = connect(config.db_path)
    scan_id = conn.execute("SELECT id FROM scans").fetchone()["id"]
    conn.close()

    resp = client.get(f"/scans/{scan_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["stage"] == "exact"
    assert body["status"] == "done"


def test_sync_without_credentials_shows_error(client, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)  # no credentials.json / token.json here

    resp = client.post("/scans/sync")

    assert resp.status_code == 200
    assert "credentials.json not found" in resp.text


def test_unknown_stage_404(client) -> None:
    assert client.post("/scans/nope").status_code == 404


def test_unknown_group_and_photo_404(client) -> None:
    assert client.get("/groups/999").status_code == 404
    assert client.get("/thumb/999").status_code == 404


def test_group_list_pagination(client, config) -> None:
    conn = connect(config.db_path)
    for i in range(25):
        insert_photo(conn, f"x{i}a", md5=f"md5-{i}")
        insert_photo(conn, f"x{i}b", md5=f"md5-{i}")
    conn.close()
    run_exact_via_ui(client)

    page1 = client.get("/groups", params={"tier": "exact", "page": 1})
    page2 = client.get("/groups", params={"tier": "exact", "page": 2})
    assert "page 1 / 2" in page1.text
    assert "page 2 / 2" in page2.text
