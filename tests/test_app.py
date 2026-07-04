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
    assert "Exact" in resp.text  # the tier navigation lists each category


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
    # exactly one exact group: the Exact tier badge in the left nav reads 1
    assert '<span class="tier-count" id="tier-count-exact">1</span>' in resp.text


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


def test_run_full_scan_chains_pipeline(config, tmp_path, monkeypatch) -> None:
    """One 'Run full scan' click runs sync -> exact -> near -> similar."""
    import numpy as np

    from tests.fakes import FakeDriveClient, FakeEmbedder, make_file

    monkeypatch.chdir(tmp_path)
    (tmp_path / "token.json").write_text("{}")  # Drive connected (OAuth)
    files = [make_file("a", md5="dup", size=2000), make_file("b", md5="dup", size=1000)]
    vec = np.random.default_rng(1).normal(size=512).astype(np.float32)
    vec /= np.linalg.norm(vec)

    app = create_app(
        config=config,
        drive_client_factory=lambda: FakeDriveClient(files),
        fetcher_factory=lambda c: FakeImageFetcher(c.cache_dir),
        embedder_factory=lambda c: FakeEmbedder({"a": vec, "b": vec}),
    )
    with TestClient(app) as ui:
        resp = ui.post("/scans/all")
        assert resp.status_code == 200
        ui.app = app
        app.state.runner.wait(timeout=30)

    conn = connect(config.db_path)
    done = {
        r["stage"]
        for r in conn.execute("SELECT DISTINCT stage FROM scans WHERE status = 'done'")
    }
    n_exact = conn.execute(
        "SELECT COUNT(*) AS n FROM groups WHERE tier = 'exact'"
    ).fetchone()["n"]
    conn.close()
    assert {"sync", "exact", "near", "similar"} <= done  # whole pipeline ran
    assert n_exact == 1  # the byte-identical pair was grouped


def test_full_scan_stops_when_a_stage_fails(config, tmp_path, monkeypatch) -> None:
    from doppel.drive import CredentialsRequired

    monkeypatch.chdir(tmp_path)
    (tmp_path / "token.json").write_text("{}")

    def boom():
        raise CredentialsRequired("drive unavailable")

    app = create_app(
        config=config,
        drive_client_factory=boom,
        fetcher_factory=lambda c: FakeImageFetcher(c.cache_dir),
    )
    with TestClient(app) as ui:
        ui.post("/scans/all")
        app.state.runner.wait(timeout=10)

    conn = connect(config.db_path)
    stages = {
        r["stage"]: r["status"] for r in conn.execute("SELECT stage, status FROM scans")
    }
    conn.close()
    assert stages.get("sync") == "failed"
    assert "exact" not in stages  # pipeline stopped after sync failed


def test_full_scan_requires_drive_connected(client, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)  # no token.json / service_account.json
    resp = client.post("/scans/all")
    assert resp.status_code == 200
    assert "not connected" in resp.text


def test_scan_timing_helper():
    from datetime import UTC, datetime, timedelta

    from doppel.app import _scan_timing

    started = (datetime.now(UTC) - timedelta(seconds=60)).isoformat(timespec="seconds")
    running = {
        "started_at": started,
        "finished_at": None,
        "status": "running",
        "processed": 25,
        "total": 100,
    }
    elapsed, eta = _scan_timing(running)
    assert elapsed is not None  # ~1m elapsed
    assert eta is not None  # 25/60s rate -> ~75 left -> a few minutes

    done = {
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "done",
        "processed": 100,
        "total": 100,
    }
    e2, eta2 = _scan_timing(done)
    assert e2 is not None and eta2 is None  # elapsed but no ETA once finished

    assert _scan_timing(None) == (None, None)
    assert _scan_timing({"started_at": None}) == (None, None)


def test_dashboard_shows_elapsed_for_running_scan(client, config):
    from datetime import UTC, datetime, timedelta

    started = (datetime.now(UTC) - timedelta(seconds=90)).isoformat(timespec="seconds")
    conn = connect(config.db_path)
    conn.execute(
        "INSERT INTO scans (stage, status, processed, total, started_at) "
        "VALUES ('near', 'running', 30, 120, ?)",
        (started,),
    )
    conn.commit()
    conn.close()

    page = client.get("/")
    assert "elapsed" in page.text
    assert "left" in page.text  # ETA rendered while running


def test_scan_is_due_logic():
    from datetime import UTC, datetime, timedelta

    from doppel.app import scan_is_due

    now = datetime.now(UTC)
    assert scan_is_due(False, None, now) is False  # disabled -> never
    assert scan_is_due(True, None, now) is True  # enabled, never scanned
    assert scan_is_due(True, now - timedelta(hours=25), now) is True  # stale
    assert scan_is_due(True, now - timedelta(hours=2), now) is False  # recent


def test_scans_partial_pushes_live_counts(client, config):
    seed_duplicates(config)
    run_exact_via_ui(client)
    resp = client.get("/partials/scans")
    assert resp.status_code == 200
    # the 2s poll carries out-of-band count spans so the left nav updates in
    # real time as the scan finds groups — no page reload needed
    assert 'id="tier-count-exact" hx-swap-oob="true"' in resp.text
    assert 'id="lib-photo-count" hx-swap-oob="true"' in resp.text


def test_resume_banner_after_interruption_and_cleared_by_clean_run(client, config):
    from doppel.jobs import now

    conn = connect(config.db_path)
    conn.execute(
        "INSERT INTO scans (stage, status, error, processed, started_at) "
        "VALUES ('near', 'failed', 'interrupted', 5, ?)",
        (now(),),
    )
    conn.commit()
    conn.close()
    page = client.get("/").text
    assert "Last scan was interrupted" in page and "Resume scan" in page

    # a later successful run of that stage supersedes it -> banner gone
    conn = connect(config.db_path)
    conn.execute(
        "INSERT INTO scans (stage, status, processed, total, started_at, finished_at) "
        "VALUES ('near', 'done', 5, 5, ?, ?)",
        (now(), now()),
    )
    conn.commit()
    conn.close()
    assert "Last scan was interrupted" not in client.get("/").text


def test_daily_scan_toggle_persists(client, config):
    from doppel.db import connect as db_connect
    from doppel.db import get_meta

    # off by default; the dashboard shows the off state
    assert "auto-scan daily: off" in client.get("/").text

    resp = client.post("/schedule", follow_redirects=False)
    assert resp.status_code == 303
    conn = db_connect(config.db_path)
    assert get_meta(conn, "daily_scan") == "on"
    conn.close()
    assert "auto-scan daily: on" in client.get("/").text

    client.post("/schedule")  # toggle back off
    conn = db_connect(config.db_path)
    assert get_meta(conn, "daily_scan") == "off"
    conn.close()
