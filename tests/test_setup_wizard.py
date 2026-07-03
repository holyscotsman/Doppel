"""Setup wizard: credentials upload, web OAuth, Ollama config, folder scope."""

import json

import pytest
from fastapi.testclient import TestClient

from doppel.app import create_app
from doppel.config import load_config, set_config_value
from doppel.db import connect
from doppel.drive import collect_folder_tree, parse_folder_input
from doppel.jobs import run_sync
from tests.fakes import FakeDriveClient, FakeImageFetcher, make_file

CONFIG_TEMPLATE = """\
thumb_size = 512
near_hamming_max = 8
dhash_confirm_max = 10
similar_cosine_min = 0.92
color_variant_min_delta = 0.25  # normalized histogram distance
clip_model = "ViT-B-32/laion2b_s34b_b79k"
db_path = "{db}"
cache_dir = "{cache}"
drive_folder_id = ""

[ollama]
host = "http://127.0.0.1:11434"
model = "test-model"
adjudicate_band_min = 0.85
brand_review_max_confidence = 0.6
"""

VALID_CLIENT_JSON = json.dumps(
    {
        "installed": {
            "client_id": "x",
            "client_secret": "y",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }
)


class FakeCredentialsObj:
    def to_json(self) -> str:
        return json.dumps({"token": "fake"})


class FakeFlow:
    def __init__(self) -> None:
        self.fetched: list[str] = []
        self.credentials = FakeCredentialsObj()

    def authorization_url(self, **kwargs):
        return "https://accounts.google.com/o/oauth2/fake", "state-abc"

    def fetch_token(self, code: str) -> None:
        self.fetched.append(code)


FOLDER_MIME = "application/vnd.google-apps.folder"

# a small Drive folder tree for the browser: id -> (name, parent, [child ids])
BROWSE_TREE = {
    "root": ("My Drive", None, ["photos_folder", "docs_folder"]),
    "photos_folder": ("Photos", "root", ["y2024_folder"]),
    "y2024_folder": ("2024", "photos_folder", []),
    "docs_folder": ("Docs", "root", []),
    "folderid1234": ("Trip", "root", []),  # used by the save-scope tests
}


SA_KEY = json.dumps(
    {
        "type": "service_account",
        "client_email": "doppel@proj.iam.gserviceaccount.com",
        "private_key": "-----BEGIN PRIVATE KEY-----\nx\n-----END PRIVATE KEY-----\n",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
)


class FakeBrowseClient:
    """Fake Drive client supporting folder navigation for the picker."""

    def __init__(self, tree: dict, shared: list[str] | None = None) -> None:
        self.tree = tree
        self.shared = shared or []  # folder ids shared with the service account

    def get_folder(self, folder_id: str) -> dict:
        if folder_id not in self.tree:
            raise RuntimeError("404 folder not found")
        name, parent, _ = self.tree[folder_id]
        meta = {"id": folder_id, "name": name, "mimeType": FOLDER_MIME}
        if parent is not None:
            meta["parents"] = [parent]
        return meta

    def list_child_folders(self, folder_id: str) -> list[dict]:
        _, _, children = self.tree[folder_id]
        return [{"id": c, "name": self.tree[c][0]} for c in children]

    def list_shared_folders(self) -> list[dict]:
        return [{"id": c, "name": self.tree[c][0]} for c in self.shared]


@pytest.fixture
def wizard(tmp_path, monkeypatch):
    """App loaded from a real config file in an isolated cwd."""
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        CONFIG_TEMPLATE.format(db=tmp_path / "t.db", cache=tmp_path / "cache")
    )
    flow = FakeFlow()
    app = create_app(
        config_path=config_file,
        fetcher_factory=lambda cfg: FakeImageFetcher(cfg.cache_dir),
        oauth_flow_factory=lambda redirect_uri: flow,
        ollama_lister=lambda host: (
            ["gemma3:27b", "qwen3.5:4b"]
            if "11434" in host
            else (_ for _ in ()).throw(ConnectionError("refused"))
        ),
        drive_client_factory=lambda: FakeBrowseClient(BROWSE_TREE),
    )
    with TestClient(app) as client:
        client.app = app
        client.flow = flow
        client.config_file = config_file
        yield client


def test_setup_page_shows_disconnected_state(wizard) -> None:
    page = wizard.get("/setup")
    assert page.status_code == 200
    assert "not connected" in page.text  # Drive not connected yet
    # a plain load must NOT probe Ollama (would stall on a slow host); the
    # dropdown appears only after an explicit test
    assert "gemma3:27b" not in page.text
    assert "test-model" in page.text  # the saved model is shown instead


def test_dashboard_banners_until_authorized(wizard, tmp_path) -> None:
    assert "setup wizard" in wizard.get("/").text

    (tmp_path / "token.json").write_text("{}")
    assert "setup wizard" not in wizard.get("/").text


def test_credentials_upload_validates_and_saves(wizard, tmp_path) -> None:
    bad = wizard.post(
        "/setup/credentials",
        files={"credentials": ("c.json", b"not json", "application/json")},
        follow_redirects=False,
    )
    assert bad.status_code == 303
    assert "err=" in bad.headers["location"]
    assert not (tmp_path / "credentials.json").exists()

    good = wizard.post(
        "/setup/credentials",
        files={
            "credentials": ("c.json", VALID_CLIENT_JSON.encode(), "application/json")
        },
        follow_redirects=False,
    )
    assert good.status_code == 303
    assert "msg=" in good.headers["location"]
    assert (tmp_path / "credentials.json").read_text() == VALID_CLIENT_JSON


def test_oauth_flow_end_to_end(wizard, tmp_path) -> None:
    # without credentials: redirected back with an error
    start = wizard.post("/oauth/start", follow_redirects=False)
    assert "err=" in start.headers["location"]

    (tmp_path / "credentials.json").write_text(VALID_CLIENT_JSON)
    start = wizard.post("/oauth/start", follow_redirects=False)
    assert start.status_code == 303
    assert start.headers["location"].startswith("https://accounts.google.com/")

    # forged/stale state is rejected — redirected back to the wizard
    bad = wizard.get("/oauth/callback?state=wrong&code=abc", follow_redirects=False)
    assert bad.status_code == 303
    assert "/setup?err=" in bad.headers["location"]
    assert not (tmp_path / "token.json").exists()  # no token written

    done = wizard.get(
        "/oauth/callback?state=state-abc&code=auth-code-1", follow_redirects=False
    )
    assert done.status_code == 303
    assert "authorized" in done.headers["location"]
    assert wizard.flow.fetched == ["auth-code-1"]
    assert json.loads((tmp_path / "token.json").read_text()) == {"token": "fake"}


def test_ollama_save_updates_config_and_reloads(wizard) -> None:
    resp = wizard.post(
        "/setup/ollama",
        data={"host": "http://127.0.0.1:11434", "model": "gemma3:27b"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "msg=" in resp.headers["location"]
    text = wizard.config_file.read_text()
    assert 'model = "gemma3:27b"' in text
    assert wizard.app.state.config.ollama.model == "gemma3:27b"
    assert wizard.app.state.vlm is None  # rebuilt with new settings


def test_ollama_save_rejects_unknown_model_and_bad_host(wizard) -> None:
    unknown = wizard.post(
        "/setup/ollama",
        data={"host": "http://127.0.0.1:11434", "model": "nope"},
        follow_redirects=False,
    )
    assert "err=" in unknown.headers["location"]

    unreachable = wizard.post(
        "/setup/ollama",
        data={"host": "http://127.0.0.1:9999", "model": "gemma3:27b"},
        follow_redirects=False,
    )
    assert "err=" in unreachable.headers["location"]
    assert wizard.app.state.config.ollama.model == "test-model"  # unchanged


def test_folder_scope_saved_after_validation(wizard) -> None:
    resp = wizard.post(
        "/setup/folder",
        data={"folder": "https://drive.google.com/drive/folders/folderid1234?usp=x"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "msg=" in resp.headers["location"]
    assert wizard.app.state.config.drive_folder_id == "folderid1234"
    assert 'drive_folder_id = "folderid1234"' in wizard.config_file.read_text()

    # back to whole Drive
    resp = wizard.post("/setup/folder", data={"folder": ""}, follow_redirects=False)
    assert wizard.app.state.config.drive_folder_id == ""


def test_folder_scope_rejects_garbage_and_unreachable(wizard) -> None:
    bad = wizard.post("/setup/folder", data={"folder": "!!!"}, follow_redirects=False)
    assert "err=" in bad.headers["location"]

    missing = wizard.post(
        "/setup/folder", data={"folder": "unknownfolder99"}, follow_redirects=False
    )
    assert "err=" in missing.headers["location"]
    assert wizard.app.state.config.drive_folder_id == ""


def test_parse_folder_input() -> None:
    assert parse_folder_input("") is None
    assert parse_folder_input("  ") is None
    assert (
        parse_folder_input("https://drive.google.com/drive/folders/abc123XYZ_-45?x=1")
        == "abc123XYZ_-45"
    )
    assert parse_folder_input("abc123XYZ_-45") == "abc123XYZ_-45"
    with pytest.raises(ValueError):
        parse_folder_input("not a folder!!")


def test_collect_folder_tree_walks_recursively() -> None:
    client = FakeDriveClient(
        [], folders={"sub1": "root12345678", "sub2": "sub1", "other": "elsewhere123"}
    )
    assert collect_folder_tree(client, "root12345678") == [
        "root12345678",
        "sub1",
        "sub2",
    ]


def test_folder_scoped_sync(conn, tmp_path) -> None:
    files = [
        make_file("in-root", md5="a", parent="root12345678"),
        make_file("in-sub", md5="b", parent="sub1"),
        make_file("outside", md5="c", parent="elsewhere123"),
    ]
    folders = {"sub1": "root12345678", "other": "elsewhere123"}

    run_sync(
        conn,
        FakeDriveClient(files, folders=folders),
        tmp_path / "cache",
        folder_ids=["root12345678"],
    )

    rows = {
        r["drive_id"]: r["status"]
        for r in conn.execute("SELECT drive_id, status FROM photos")
    }
    assert rows == {"in-root": "active", "in-sub": "active"}

    # a previously full sync photo outside the scope gets marked missing
    run_sync(conn, FakeDriveClient(files, folders=folders), tmp_path / "cache")
    run_sync(
        conn,
        FakeDriveClient(files, folders=folders),
        tmp_path / "cache",
        folder_ids=["root12345678"],
    )
    rows = {
        r["drive_id"]: r["status"]
        for r in conn.execute("SELECT drive_id, status FROM photos")
    }
    assert rows["outside"] == "missing"
    assert rows["in-root"] == "active"


def test_set_config_value_preserves_comments(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        'thumb_size = 512\ndrive_folder_id = ""  # empty = whole Drive\n'
        '\n[ollama]\nhost = "http://a"\nmodel = "old"\n'
    )

    set_config_value(path, "drive_folder_id", "xyz")
    set_config_value(path, "model", "new-model", section="ollama")

    text = path.read_text()
    assert 'drive_folder_id = "xyz"  # empty = whole Drive' in text
    assert 'model = "new-model"' in text
    assert 'host = "http://a"' in text
    assert "thumb_size = 512" in text
    cfg_text = text  # sanity: still parseable TOML
    import tomllib

    parsed = tomllib.loads(cfg_text)
    assert parsed["drive_folder_id"] == "xyz"
    assert parsed["ollama"]["model"] == "new-model"


def test_set_config_value_inserts_missing_key(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('thumb_size = 512\n\n[ollama]\nhost = "http://a"\n')

    set_config_value(path, "drive_folder_id", "abc")

    import tomllib

    parsed = tomllib.loads(path.read_text())
    assert parsed["drive_folder_id"] == "abc"
    assert parsed["thumb_size"] == 512
    assert parsed["ollama"]["host"] == "http://a"


def test_scoped_sync_runs_through_the_dashboard_button(tmp_path, monkeypatch) -> None:
    """The wizard-set folder scope actually reaches run_sync when the sync
    stage is launched from the UI."""
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        CONFIG_TEMPLATE.format(db=tmp_path / "t.db", cache=tmp_path / "cache").replace(
            'drive_folder_id = ""', 'drive_folder_id = "root12345678"'
        )
    )
    (tmp_path / "token.json").write_text("{}")  # sync precheck: Drive connected
    files = [
        make_file("in-scope", md5="a", parent="root12345678"),
        make_file("out-scope", md5="b", parent="elsewhere123"),
    ]
    fake_drive = FakeDriveClient(files, folders={"other": "elsewhere123"})
    app = create_app(
        config_path=config_file,
        fetcher_factory=lambda cfg: FakeImageFetcher(cfg.cache_dir),
        drive_client_factory=lambda: fake_drive,
    )
    with TestClient(app) as client:
        resp = client.post("/scans/sync")
        assert resp.status_code == 200
        app.state.runner.wait(timeout=10)

    conn = connect(tmp_path / "t.db")
    rows = {r["drive_id"] for r in conn.execute("SELECT drive_id FROM photos")}
    conn.close()
    assert rows == {"in-scope"}  # the out-of-scope photo was never synced


def test_same_origin_guard_blocks_cross_origin_writes(wizard) -> None:
    # a browser attaches Origin to cross-origin POSTs; forge one
    forged = wizard.post(
        "/setup/folder",
        data={"folder": ""},
        headers={"origin": "https://evil.example"},
        follow_redirects=False,
    )
    assert forged.status_code == 403

    # same-origin (Origin matches the test host) is allowed
    ok = wizard.post(
        "/setup/folder",
        data={"folder": ""},
        headers={"origin": "http://testserver"},
        follow_redirects=False,
    )
    assert ok.status_code == 303

    # no Origin/Referer (curl, the test suite) is allowed through
    plain = wizard.post("/setup/folder", data={"folder": ""}, follow_redirects=False)
    assert plain.status_code == 303


def test_oauth_start_requires_post(wizard, tmp_path) -> None:
    (tmp_path / "credentials.json").write_text(VALID_CLIENT_JSON)
    assert wizard.get("/oauth/start").status_code == 405  # GET no longer routed


def test_oauth_callback_lost_state_returns_to_wizard(wizard) -> None:
    # state lost (server restart / double-start): redirect back, not a 400
    resp = wizard.get("/oauth/callback?state=stale&code=abc", follow_redirects=False)
    assert resp.status_code == 303
    assert "/setup?err=" in resp.headers["location"]


def test_set_config_value_escapes_quotes(tmp_path) -> None:
    import tomllib

    path = tmp_path / "config.toml"
    path.write_text('[ollama]\nmodel = "old"\n')
    set_config_value(path, "model", 'weird"name\nwith-newline', section="ollama")
    # file stays valid TOML and round-trips exactly
    parsed = tomllib.loads(path.read_text())
    assert parsed["ollama"]["model"] == 'weird"name\nwith-newline'


def test_set_config_value_matches_indented_key(tmp_path) -> None:
    import tomllib

    path = tmp_path / "config.toml"
    path.write_text('[ollama]\n  model = "old"\n')  # indented, still valid TOML
    set_config_value(path, "model", "new", section="ollama")
    parsed = tomllib.loads(path.read_text())
    assert parsed["ollama"]["model"] == "new"  # replaced, not duplicated


def test_ollama_model_with_quote_does_not_brick_config(wizard) -> None:
    # a malicious/odd host advertising a quote-laden model name must not
    # corrupt config.toml
    resp = wizard.post(
        "/setup/ollama",
        data={"host": "http://127.0.0.1:11434", "model": 'gemma"3'},
        follow_redirects=False,
    )
    # rejected (not in the fake's model list) OR saved-but-valid; either way
    # the config still loads
    assert resp.status_code == 303
    load_config(wizard.config_file)  # raises if corrupted


# --- Ollama: test-connection feedback + usable-model filtering ---


class FakeShow:
    def __init__(self, capabilities):
        self.capabilities = capabilities


class FakeOllamaClient:
    """Mimics the ollama Client: .list() and .show(name).capabilities."""

    def __init__(self, models):
        # models: dict name -> capabilities list
        self._models = models

    def list(self):
        return {"models": [{"model": n} for n in self._models]}

    def show(self, name):
        return FakeShow(self._models[name])


def test_usable_models_filters_vision_and_denylist():
    from doppel.app import _usable_models

    client = FakeOllamaClient(
        {
            "gemma3:27b": ["completion", "vision"],  # keep
            "qwen3-vl:latest": ["completion", "vision"],  # keep
            "minicpm-v:latest": ["completion", "vision"],  # drop (denylist)
            "llava:13b": ["completion", "vision"],  # drop (denylist)
            "llama3:8b": ["completion"],  # drop (no vision)
        }
    )
    assert _usable_models(client) == ["gemma3:27b", "qwen3-vl:latest"]


def test_test_connection_reports_success(wizard):
    page = wizard.get("/setup", params={"ollama_host": "http://127.0.0.1:11434"})
    assert "Test successful" in page.text
    assert "2 usable models" in page.text
    assert "gemma3:27b" in page.text  # dropdown populated


def test_test_connection_reports_failure(wizard):
    page = wizard.get("/setup", params={"ollama_host": "http://127.0.0.1:9999"})
    assert "Test failed" in page.text
    assert "gemma3:27b" not in page.text  # no dropdown when unreachable


def test_test_connection_reports_no_usable_models(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        CONFIG_TEMPLATE.format(db=tmp_path / "t.db", cache=tmp_path / "cache")
    )
    app = create_app(
        config_path=cfg,
        fetcher_factory=lambda c: FakeImageFetcher(c.cache_dir),
        ollama_lister=lambda host: [],  # reachable, but nothing usable
    )
    with TestClient(app) as client:
        page = client.get("/setup", params={"ollama_host": "http://127.0.0.1:11434"})
        assert "no multi-image vision models" in page.text.lower()


# --- Drive folder browser ---


def test_setup_shows_browser_only_when_authorized(wizard, tmp_path):
    # not authorized yet: prompt to connect first
    page = wizard.get("/setup")
    assert "Connect Google Drive first" in page.text
    assert 'hx-get="/drive/browse' not in page.text

    (tmp_path / "token.json").write_text("{}")
    page = wizard.get("/setup")
    assert "/drive/browse?folder=root" in page.text  # OAuth entry = My Drive


def test_drive_browse_lists_my_drive(wizard):
    page = wizard.get("/drive/browse", params={"folder": "root", "home": "root"})
    assert page.status_code == 200
    assert "My Drive" in page.text
    assert "Photos" in page.text and "Docs" in page.text  # top-level folders
    assert "Scan entire Drive" in page.text


def test_drive_browse_drills_in_with_breadcrumb(wizard):
    page = wizard.get(
        "/drive/browse", params={"folder": "photos_folder", "home": "root"}
    )
    assert "Photos" in page.text
    assert "2024" in page.text  # subfolder shown
    assert "up" in page.text  # up-nav present
    assert "Scan “Photos”" in page.text
    assert "/drive/browse?folder=y2024_folder" in page.text


def test_drive_browse_requires_auth():
    from doppel.drive import CredentialsRequired

    def needs_auth():
        raise CredentialsRequired("authorize first")

    app = create_app(
        config=None,
        config_path="config.toml",  # real config only for defaults; not written
        fetcher_factory=lambda c: FakeImageFetcher(c.cache_dir),
        drive_client_factory=needs_auth,
    )
    with TestClient(app) as client:
        page = client.get("/drive/browse", params={"folder": "root"})
        assert "Connect Google Drive first" in page.text


def test_browser_select_saves_folder_scope(wizard):
    resp = wizard.post(
        "/setup/folder", data={"folder": "folderid1234"}, follow_redirects=False
    )
    assert resp.status_code == 303
    assert "msg=" in resp.headers["location"]
    assert wizard.app.state.config.drive_folder_id == "folderid1234"


def test_browser_scan_entire_drive_clears_scope(wizard):
    wizard.post("/setup/folder", data={"folder": "folderid1234"})
    resp = wizard.post("/setup/folder", data={"folder": ""}, follow_redirects=False)
    assert resp.status_code == 303
    assert wizard.app.state.config.drive_folder_id == ""


# --- Drive auth simplification ---


def test_signin_button_shown_when_client_present(wizard, tmp_path):
    (tmp_path / "credentials.json").write_text(VALID_CLIENT_JSON)
    page = wizard.get("/setup")
    assert 'action="/oauth/start"' in page.text  # the sign-in button
    assert 'action="/setup/credentials"' not in page.text  # upload hidden


def test_upload_shown_only_before_client_present(wizard):
    page = wizard.get("/setup")  # no credentials.json in this tmp cwd
    assert 'action="/setup/credentials"' in page.text  # upload form
    assert 'action="/oauth/start"' not in page.text  # no sign-in until configured


def test_plain_setup_load_does_not_probe_ollama(tmp_path, monkeypatch):
    """Auto-opened /setup must not block on Ollama; probe only on test."""
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        CONFIG_TEMPLATE.format(db=tmp_path / "t.db", cache=tmp_path / "cache")
    )
    calls: list[str] = []

    def counting_lister(host):
        calls.append(host)
        return ["gemma3:27b"]

    app = create_app(
        config_path=cfg,
        fetcher_factory=lambda c: FakeImageFetcher(c.cache_dir),
        ollama_lister=counting_lister,
    )
    with TestClient(app) as client:
        client.get("/setup")  # plain load
        assert calls == []  # no probe
        client.get("/setup", params={"ollama_host": "http://127.0.0.1:11434"})
        assert calls == ["http://127.0.0.1:11434"]  # probe only on test


def test_failed_test_shows_ollama_install_help(wizard):
    # host that the fixture lister rejects -> unreachable -> install steps
    page = wizard.get("/setup", params={"ollama_host": "http://127.0.0.1:9999"})
    assert "Test failed" in page.text
    assert "Get Ollama running" in page.text
    assert "macOS" in page.text and "Windows" in page.text
    assert "ollama.com/download" in page.text


def test_no_models_shows_pull_hint_not_install(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        CONFIG_TEMPLATE.format(db=tmp_path / "t.db", cache=tmp_path / "cache")
    )
    app = create_app(
        config_path=cfg,
        fetcher_factory=lambda c: FakeImageFetcher(c.cache_dir),
        ollama_lister=lambda host: [],  # reachable, no usable models
    )
    with TestClient(app) as client:
        page = client.get("/setup", params={"ollama_host": "http://127.0.0.1:11434"})
        assert "ollama pull gemma3" in page.text
        assert "Get Ollama running" not in page.text  # it IS running


# --- Service-account auth (the recommended, no-consent-screen path) ---


def test_upload_service_account_key_connects(wizard, tmp_path):
    resp = wizard.post(
        "/setup/credentials",
        files={"credentials": ("sa.json", SA_KEY.encode(), "application/json")},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "msg=" in resp.headers["location"]
    # routed to service_account.json, NOT credentials.json
    assert (tmp_path / "service_account.json").exists()
    assert not (tmp_path / "credentials.json").exists()
    # the redirect tells the user which email to share their folder with
    assert "gserviceaccount.com" in resp.headers["location"]


def test_service_account_mode_setup_page(wizard, tmp_path):
    (tmp_path / "service_account.json").write_text(SA_KEY)
    page = wizard.get("/setup")
    assert "connected (service account)" in page.text
    assert "doppel@proj.iam.gserviceaccount.com" in page.text  # share-with email
    assert "Share" in page.text
    # scope browser starts from shared-with-me, not My Drive
    assert "/drive/browse?folder=shared" in page.text


def test_auth_mode_prefers_service_account(wizard, tmp_path):
    from doppel.app import auth_mode

    (tmp_path / "token.json").write_text("{}")
    assert auth_mode() == "oauth"
    (tmp_path / "service_account.json").write_text(SA_KEY)
    assert auth_mode() == "service_account"  # SA takes precedence


def test_browse_shared_lists_shared_folders(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        CONFIG_TEMPLATE.format(db=tmp_path / "t.db", cache=tmp_path / "cache")
    )
    (tmp_path / "service_account.json").write_text(SA_KEY)
    app = create_app(
        config_path=cfg,
        fetcher_factory=lambda c: FakeImageFetcher(c.cache_dir),
        drive_client_factory=lambda: FakeBrowseClient(
            BROWSE_TREE, shared=["photos_folder"]
        ),
    )
    with TestClient(app) as client:
        page = client.get(
            "/drive/browse", params={"folder": "shared", "home": "shared"}
        )
        assert "Photos" in page.text  # the shared folder
        assert "Scan entire Drive" not in page.text  # no "everything" in SA mode
        # drilling into the shared folder is offered
        assert "/drive/browse?folder=photos_folder" in page.text


def test_service_account_sync_via_ui(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        CONFIG_TEMPLATE.format(db=tmp_path / "t.db", cache=tmp_path / "cache").replace(
            'drive_folder_id = ""', 'drive_folder_id = "photos_folder"'
        )
    )
    (tmp_path / "service_account.json").write_text(SA_KEY)
    files = [
        make_file("p1", md5="a", parent="photos_folder"),
        make_file("p2", md5="b", parent="elsewhere"),
    ]

    class SyncClient(FakeDriveClient):
        def list_folders_page(self, parent_id, page_token=None):
            return {"files": []}  # photos_folder has no subfolders

    fake = SyncClient(files, folders={})
    app = create_app(
        config_path=cfg,
        fetcher_factory=lambda c: FakeImageFetcher(c.cache_dir),
        drive_client_factory=lambda: fake,
    )
    with TestClient(app) as client:
        resp = client.post("/scans/sync")  # SA mode counts as connected
        assert resp.status_code == 200
        app.state.runner.wait(timeout=10)

    conn = connect(tmp_path / "t.db")
    rows = {r["drive_id"] for r in conn.execute("SELECT drive_id FROM photos")}
    conn.close()
    assert rows == {"p1"}  # only the shared folder's file was synced


# --- Fixes from the service-account review ---


def test_service_account_empty_scope_scans_shared_folders(tmp_path, monkeypatch):
    """SA mode + no folder picked must scan the shared folders, not run an
    unscoped My-Drive query (which is empty for a service account and would
    wrongly mark the whole inventory missing)."""
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        CONFIG_TEMPLATE.format(db=tmp_path / "t.db", cache=tmp_path / "cache")
    )  # drive_folder_id = "" (no explicit scope)
    (tmp_path / "service_account.json").write_text(SA_KEY)
    files = [
        make_file("shared1", md5="a", parent="photos_folder"),
        make_file("elsewhere", md5="b", parent="not_shared"),
    ]

    class SAClient(FakeDriveClient):
        def list_shared_folders(self):
            return [{"id": "photos_folder", "name": "Photos"}]

        def list_folders_page(self, parent_id, page_token=None):
            return {"files": []}

    app = create_app(
        config_path=cfg,
        fetcher_factory=lambda c: FakeImageFetcher(c.cache_dir),
        drive_client_factory=lambda: SAClient(files, folders={}),
    )
    with TestClient(app) as client:
        client.post("/scans/sync")
        app.state.runner.wait(timeout=10)

    conn = connect(tmp_path / "t.db")
    rows = {r["drive_id"] for r in conn.execute("SELECT drive_id FROM photos")}
    conn.close()
    assert rows == {"shared1"}  # only the shared folder, not the whole world


def test_service_account_nothing_shared_errors_not_wipes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        CONFIG_TEMPLATE.format(db=tmp_path / "t.db", cache=tmp_path / "cache")
    )
    (tmp_path / "service_account.json").write_text(SA_KEY)
    # a pre-existing photo that must NOT be wiped to 'missing'
    conn = connect(tmp_path / "t.db")
    conn.execute(
        "INSERT INTO photos (drive_id, name, mime_type, status) "
        "VALUES ('old', 'o.jpg', 'image/jpeg', 'active')"
    )
    conn.commit()
    conn.close()

    class NoShareClient(FakeDriveClient):
        def list_shared_folders(self):
            return []

    app = create_app(
        config_path=cfg,
        fetcher_factory=lambda c: FakeImageFetcher(c.cache_dir),
        drive_client_factory=lambda: NoShareClient([], folders={}),
    )
    with TestClient(app) as client:
        client.post("/scans/sync")
        app.state.runner.wait(timeout=10)

    conn = connect(tmp_path / "t.db")
    scan = conn.execute(
        "SELECT status, error FROM scans WHERE stage='sync' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    status = conn.execute("SELECT status FROM photos WHERE drive_id='old'").fetchone()
    conn.close()
    assert scan["status"] == "failed"
    assert "shared" in scan["error"].lower()
    assert status["status"] == "active"  # inventory NOT wiped


def test_browse_up_to_real_root_offers_entire_drive(wizard):
    """Real Drive returns My Drive's actual id (not 'root'); 'up' to it must
    still offer 'Scan entire Drive', keyed on parentlessness not a string."""
    real_root = "0ABCrealrootid"
    tree = {
        real_root: ("My Drive", None, ["photos_x"]),
        "photos_x": ("Photos", real_root, []),
    }
    app = create_app(
        config_path=wizard.config_file,
        fetcher_factory=lambda c: FakeImageFetcher(c.cache_dir),
        drive_client_factory=lambda: FakeBrowseClient(tree),
    )
    with TestClient(app) as client:
        # 'up' navigates to the real root id (has no parent)
        page = client.get("/drive/browse", params={"folder": real_root, "home": "root"})
        assert "Scan entire Drive" in page.text
        assert "current scope" in page.text  # empty scope == entire Drive


def test_disconnect_removes_active_credential(wizard, tmp_path):
    (tmp_path / "service_account.json").write_text(SA_KEY)
    (tmp_path / "token.json").write_text("{}")
    # SA takes precedence; disconnect should drop it and fall back to OAuth
    resp = wizard.post("/setup/disconnect", follow_redirects=False)
    assert resp.status_code == 303
    assert not (tmp_path / "service_account.json").exists()
    from doppel.app import auth_mode

    assert auth_mode() == "oauth"  # recovered, not stuck

    # disconnecting again drops the OAuth token too
    wizard.post("/setup/disconnect")
    assert not (tmp_path / "token.json").exists()
    assert auth_mode() is None
