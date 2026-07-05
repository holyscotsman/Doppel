"""Phase 4: setup persistence (#12) — the home page redirects to the wizard
until a marker file is written, and deleting it re-triggers setup — plus the
best-vision-model auto-pick ranking (#13)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from doppel.app import _best_vision_model, _rank_vision_model, create_app
from tests.fakes import FakeImageFetcher


def _client(config, marker, enforce=True):
    app = create_app(
        config=config,
        enforce_setup=enforce,
        setup_marker=marker,
        fetcher_factory=lambda cfg: FakeImageFetcher(cfg.cache_dir),
    )
    return TestClient(app)


# ---- #12 setup persistence ----------------------------------------------


def test_home_redirects_to_setup_without_marker(config, tmp_path) -> None:
    c = _client(config, tmp_path / "settings.ini")
    resp = c.get("/", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/setup"


def test_finish_setup_writes_marker_and_home_becomes_dashboard(
    config, tmp_path
) -> None:
    marker = tmp_path / "settings.ini"
    c = _client(config, marker)
    done = c.post("/setup/complete", follow_redirects=False)
    assert done.status_code == 303 and done.headers["location"] == "/"
    assert marker.exists()
    assert (
        c.get("/", follow_redirects=False).status_code == 200
    )  # dashboard, no redirect


def test_deleting_marker_re_triggers_setup(config, tmp_path) -> None:
    marker = tmp_path / "settings.ini"
    c = _client(config, marker)
    c.post("/setup/complete")
    assert marker.exists()
    marker.unlink()
    resp = c.get("/", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/setup"


def test_no_gate_when_enforce_off(config, tmp_path) -> None:
    # tests build create_app() without enforce_setup, so / stays the dashboard
    c = _client(config, tmp_path / "settings.ini", enforce=False)
    assert c.get("/", follow_redirects=False).status_code == 200


# ---- #13 best-model auto-pick -------------------------------------------


def test_best_model_prefers_a_qwen_vl() -> None:
    models = ["gemma3:12b", "qwen2.5-vl:7b", "llava:13b"]
    assert _best_vision_model(models) == "qwen2.5-vl:7b"


def test_rank_puts_qwen_vl_above_gemma() -> None:
    assert _rank_vision_model("qwen2.5-vl:7b") > _rank_vision_model("gemma3:12b")


def test_best_model_falls_back_to_first_when_all_unknown() -> None:
    assert _best_vision_model(["mystery:latest", "other:1b"]) == "mystery:latest"


def test_best_model_none_for_empty_list() -> None:
    assert _best_vision_model([]) is None
