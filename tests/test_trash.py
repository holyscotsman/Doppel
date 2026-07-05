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
    # the failure is explained in plain language, not dumped as raw Drive text
    assert "Not yours to move" in resp.text
    assert "insufficientFilePermissions" not in resp.text

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

    assert "Trash queue" in trash_client.get("/").text
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


def test_classify_trash_error_maps_permission_denied() -> None:
    from doppel.drive import classify_trash_error

    code, reason = classify_trash_error(
        RuntimeError("insufficientFilePermissions: shared read-only")
    )
    assert code == "not_owner"
    assert "owner" in reason.lower()
    # a transient rate-limit is a different, non-ownership cause
    rate = classify_trash_error(RuntimeError("userRateLimitExceeded"))
    assert rate[0] == "rate_limited"


def test_service_account_failure_explains_ownership(config, monkeypatch):
    """When every file fails under a service account, the result page explains
    the systemic cause (SA isn't the owner) once — not 22 raw error dumps."""
    import doppel.app as appmod

    monkeypatch.setattr(appmod, "auth_mode", lambda: "service_account")
    fake = FakeTrashClient(fail_ids={"a", "b"})
    app = create_app(
        config=config,
        fetcher_factory=lambda cfg: FakeImageFetcher(cfg.cache_dir),
        trash_client_factory=lambda: fake,
    )
    conn = connect(config.db_path)
    mark_trash(conn, "a", 100)
    mark_trash(conn, "b", 200)
    conn.close()

    with TestClient(app) as c:
        resp = c.post("/trash")
    assert resp.status_code == 200
    assert "No photos were moved" in resp.text
    assert "service account" in resp.text  # the systemic explanation banner
    assert "owner" in resp.text.lower()
    assert "/oauth/trash/start" in resp.text  # offers the owner sign-in


# ---- write-scoped owner sign-in (#trash OAuth) --------------------------


class _FakeWriteCreds:
    def to_json(self) -> str:
        return '{"token": "fake-write"}'


class _FakeWriteFlow:
    """Stand-in for the write-scoped Google OAuth flow."""

    def __init__(self) -> None:
        self.credentials = _FakeWriteCreds()
        self.code: str | None = None

    def authorization_url(self, **kwargs):
        return "https://accounts.google.com/o/oauth2/write", "st-write"

    def fetch_token(self, code: str) -> None:
        self.code = code


@pytest.fixture
def trash_oauth(config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "credentials.json").write_text("{}")  # presence check only
    flow = _FakeWriteFlow()
    app = create_app(
        config=config,
        fetcher_factory=lambda cfg: FakeImageFetcher(cfg.cache_dir),
        trash_oauth_flow_factory=lambda redirect_uri: flow,
    )
    with TestClient(app) as c:
        c.flow = flow
        c.tmp = tmp_path
        yield c


def test_owner_token_is_preferred_over_scanning_credential(config, monkeypatch):
    """When a write-scoped owner token is connected, the trash action uses it —
    not the service account — because only the owner can trash their files."""
    import doppel.app as appmod

    monkeypatch.setattr(appmod, "auth_mode", lambda: "service_account")
    sentinel = object()
    monkeypatch.setattr(appmod, "load_trash_oauth_credentials", lambda: sentinel)
    assert appmod.build_trash_credentials() is sentinel
    assert appmod.can_trash() is True
    assert appmod.trash_owner_connected() is True


def test_connect_trash_account_starts_write_flow(trash_oauth):
    resp = trash_oauth.post("/oauth/trash/start", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"].startswith(
        "https://accounts.google.com/o/oauth2/write"
    )


def test_trash_oauth_callback_writes_a_separate_write_token(trash_oauth):
    trash_oauth.post("/oauth/trash/start", follow_redirects=False)  # sets pending state
    resp = trash_oauth.get(
        "/oauth/callback?state=st-write&code=code-w", follow_redirects=False
    )
    assert resp.status_code == 303
    assert "/trash/confirm" in resp.headers["location"]  # back to the trash page
    # the write token is stored SEPARATELY so it never disturbs the scan token
    assert (
        trash_oauth.tmp / "token_write.json"
    ).read_text() == '{"token": "fake-write"}'
    assert not (trash_oauth.tmp / "token.json").exists()


def test_trash_start_without_credentials_sends_to_setup(config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no credentials.json present
    app = create_app(
        config=config, fetcher_factory=lambda cfg: FakeImageFetcher(cfg.cache_dir)
    )
    with TestClient(app) as c:
        resp = c.post("/oauth/trash/start", follow_redirects=False)
    assert resp.status_code == 303
    assert "/setup" in resp.headers["location"]


def test_confirm_page_offers_owner_signin_in_service_account_mode(config, monkeypatch):
    import doppel.app as appmod

    monkeypatch.setattr(appmod, "auth_mode", lambda: "service_account")
    monkeypatch.setattr(appmod, "trash_owner_connected", lambda: False)
    app = create_app(
        config=config, fetcher_factory=lambda cfg: FakeImageFetcher(cfg.cache_dir)
    )
    conn = connect(config.db_path)
    mark_trash(conn, "a", 100)
    conn.close()
    with TestClient(app) as c:
        page = c.get("/trash/confirm")
    assert "service account" in page.text
    assert "/oauth/trash/start" in page.text  # the connect button is offered
