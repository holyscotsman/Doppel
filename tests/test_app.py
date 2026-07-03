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


def test_near_stage_via_ui(client, config) -> None:
    conn = connect(config.db_path)
    insert_photo(conn, "n1", md5="m1")
    insert_photo(conn, "n2", md5="m2")
    conn.close()

    resp = client.post("/scans/near")
    assert resp.status_code == 200
    client.app.state.runner.wait(timeout=10)

    # both photos get the fake fetcher's default image -> one near group
    page = client.get("/groups", params={"tier": "near"})
    assert "2 photos" in page.text


def test_similar_stage_via_ui(config) -> None:
    import numpy as np

    from tests.fakes import FakeEmbedder, FakeImageFetcher

    conn = connect(config.db_path)
    insert_photo(conn, "s1", md5="m1")
    insert_photo(conn, "s2", md5="m2")
    conn.close()

    vec = np.random.default_rng(5).normal(size=512).astype(np.float32)
    vec /= np.linalg.norm(vec)
    app = create_app(
        config=config,
        fetcher_factory=lambda cfg: FakeImageFetcher(cfg.cache_dir),
        embedder_factory=lambda cfg: FakeEmbedder({"s1": vec, "s2": vec}),
    )
    with TestClient(app) as ui:
        resp = ui.post("/scans/similar")
        assert resp.status_code == 200
        app.state.runner.wait(timeout=10)

        page = ui.get("/groups", params={"tier": "similar"})
        assert "2 photos" in page.text


def test_dashboard_shows_exact_group_count(client, config) -> None:
    seed_duplicates(config)
    run_exact_via_ui(client)

    resp = client.get("/")
    # exactly one exact group: its count card must render 1
    assert '<div class="big">1</div>' in resp.text


def test_stage_error_is_delivered_out_of_band(client, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    resp = client.post("/scans/sync")

    # error rides an hx-swap-oob div so the 2s poller cannot erase it
    assert 'id="scan-error" hx-swap-oob="true"' in resp.text
    assert "credentials.json not found" in resp.text

    # the polled partial itself never carries the error
    partial = client.get("/partials/scans")
    assert "credentials.json" not in partial.text


def test_sync_auth_failure_recorded_in_scans_ledger(
    client, config, monkeypatch, tmp_path
) -> None:
    # token.json exists but is garbage and there is no credentials.json:
    # the precheck passes, the worker must record a failed scan
    monkeypatch.chdir(tmp_path)
    (tmp_path / "token.json").write_text("not json")

    resp = client.post("/scans/sync")
    assert resp.status_code == 200
    client.app.state.runner.wait(timeout=10)

    conn = connect(config.db_path)
    scan = conn.execute(
        "SELECT * FROM scans WHERE stage = 'sync' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert scan is not None
    assert scan["status"] == "failed"
    assert (
        "authorization required" in scan["error"].lower()
        or "CredentialsRequired" in scan["error"]
    )
