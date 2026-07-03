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
    monkeypatch.chdir(tmp_path)  # no token.json here

    resp = client.post("/scans/sync")

    assert resp.status_code == 200
    assert "not connected" in resp.text


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
    assert "not connected" in resp.text

    # the polled partial itself never carries the error
    partial = client.get("/partials/scans")
    assert "not connected" not in partial.text


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


def test_adjudicate_stage_via_ui_shows_verdict(config) -> None:
    from doppel.jobs import now as job_now
    from tests.fakes import FakeImageFetcher, FakeVlm

    conn = connect(config.db_path)
    a = insert_photo(conn, "red", name="red.jpg", md5="r")
    b = insert_photo(conn, "blue", name="blue.jpg", md5="bl")
    cur = conn.execute(
        "INSERT INTO groups (tier, color_variant, created_at) VALUES ('near', 1, ?)",
        (job_now(),),
    )
    gid = cur.lastrowid
    conn.executemany(
        "INSERT INTO group_members (group_id, photo_id) VALUES (?, ?)",
        [(gid, a), (gid, b)],
    )
    conn.commit()
    conn.close()

    app = create_app(
        config=config,
        fetcher_factory=lambda cfg: FakeImageFetcher(cfg.cache_dir),
        vlm_factory=lambda cfg: FakeVlm(
            [{"verdict": "variant", "reason": "same shirt, different color"}]
        ),
    )
    with TestClient(app) as ui:
        resp = ui.post("/scans/adjudicate")
        assert resp.status_code == 200
        app.state.runner.wait(timeout=10)

        listing = ui.get("/groups", params={"tier": "vlm"})
        assert "2 photos" in listing.text
        assert "color variant" in listing.text

        conn = connect(config.db_path)
        group_id = conn.execute("SELECT id FROM groups WHERE tier = 'vlm'").fetchone()[
            "id"
        ]
        conn.close()
        detail = ui.get(f"/groups/{group_id}")
        assert "same shirt, different color" in detail.text
        assert "variant" in detail.text


def test_brand_pages_filter_and_correction(config) -> None:
    from tests.fakes import FakeImageFetcher, FakeVlm

    conn = connect(config.db_path)
    confident = insert_photo(conn, "c1", name="jacket.jpg", md5="a")
    doubtful = insert_photo(conn, "d1", name="blurry.jpg", md5="b")
    conn.close()

    app = create_app(
        config=config,
        fetcher_factory=lambda cfg: FakeImageFetcher(cfg.cache_dir),
        vlm_factory=lambda cfg: FakeVlm(
            [
                {"brand": "Patagonia", "evidence": "chest logo", "confidence": 0.95},
                {"brand": "Nike", "evidence": "maybe a swoosh", "confidence": 0.3},
            ]
        ),
    )
    with TestClient(app) as ui:
        resp = ui.post("/scans/brand")
        assert resp.status_code == 200
        app.state.runner.wait(timeout=10)

        summary = ui.get("/brands")
        assert "Patagonia" in summary.text
        assert "Nike" in summary.text
        assert "review queue: 1 tag" in summary.text

        queue = ui.get("/brands/photos", params={"queue": 1})
        assert "blurry.jpg" in queue.text
        assert "jacket.jpg" not in queue.text

        filtered = ui.get("/brands/photos", params={"brand": "Patagonia"})
        assert "jacket.jpg" in filtered.text
        assert "blurry.jpg" not in filtered.text

        # human correction from the queue
        resp = ui.post(
            f"/photos/{doubtful}/brand",
            data={"value": "Under Armour", "queue": "1"},
            follow_redirects=False,
        )
        assert resp.status_code == 303

        conn = connect(config.db_path)
        tag = conn.execute(
            "SELECT value, source FROM tags WHERE photo_id = ?", (doubtful,)
        ).fetchone()
        conn.close()
        assert (tag["value"], tag["source"]) == ("Under Armour", "human")

        # corrected tag leaves the low-confidence queue
        queue = ui.get("/brands/photos", params={"queue": 1})
        assert "blurry.jpg" not in queue.text
    _ = confident
