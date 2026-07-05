"""Phase 6 (#14): the version + build string shown in the corner."""

from __future__ import annotations

from fastapi.testclient import TestClient

from doppel.app import _app_version, create_app
from tests.fakes import FakeImageFetcher


def test_version_string_starts_with_the_app_version() -> None:
    assert _app_version().startswith("v0.1.0")


def test_version_badge_renders_on_the_page(config) -> None:
    app = create_app(
        config=config, fetcher_factory=lambda cfg: FakeImageFetcher(cfg.cache_dir)
    )
    with TestClient(app) as c:
        page = c.get("/setup")
        assert 'class="ver-badge"' in page.text
        assert _app_version() in page.text
