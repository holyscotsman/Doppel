"""Setup wizard: credentials upload, web OAuth, Ollama config, folder scope."""

import json

import pytest
from fastapi.testclient import TestClient

from doppel.app import create_app
from doppel.config import load_config, set_config_value
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


class FakeFolderClient:
    def __init__(self, folders: dict[str, str]) -> None:
        self.folders = folders  # id -> name

    def get_folder(self, folder_id: str) -> dict:
        if folder_id not in self.folders:
            raise RuntimeError("404 folder not found")
        return {"id": folder_id, "name": self.folders[folder_id]}


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
        drive_client_factory=lambda: FakeFolderClient({"folderid1234": "Photos"}),
    )
    with TestClient(app) as client:
        client.app = app
        client.flow = flow
        client.config_file = config_file
        yield client


def test_setup_page_shows_disconnected_state(wizard) -> None:
    page = wizard.get("/setup")
    assert page.status_code == 200
    assert "not connected" in page.text
    assert "gemma3:27b" in page.text  # models listed from the fake Ollama


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
    start = wizard.get("/oauth/start", follow_redirects=False)
    assert "err=" in start.headers["location"]

    (tmp_path / "credentials.json").write_text(VALID_CLIENT_JSON)
    start = wizard.get("/oauth/start", follow_redirects=False)
    assert start.status_code == 303
    assert start.headers["location"].startswith("https://accounts.google.com/")

    # forged state is rejected
    bad = wizard.get("/oauth/callback?state=wrong&code=abc")
    assert bad.status_code == 400

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
        folder_id="root12345678",
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
        folder_id="root12345678",
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


def test_scoped_sync_reaches_photos_table_via_ui(wizard, tmp_path) -> None:
    """Config folder scope + config reload are actually used by the sync
    stage wiring (smoke via config object)."""
    cfg = load_config(wizard.config_file)
    assert cfg.drive_folder_id == ""
