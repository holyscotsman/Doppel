"""Phase 2 (#4): folder path links to the Drive folder, each photo links to its
Drive file (both new-tab), the thumbnail carries a lightbox target, and the
larger-preview route serves a size-parameterized thumbnail."""

from __future__ import annotations

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
    with TestClient(app) as c:
        c.app = app
        yield c


@pytest.fixture
def group(client, config):
    """An exact group; the big copy is filed in a real folder, the small one has
    no parent. Returns (group_id, big_id, small_id)."""
    conn = connect(config.db_path)
    big = insert_photo(
        conn,
        "aaa",
        name="a.jpg",
        md5="dup",
        size=5000,
        parent_id="folder123",
        folder_path="Photos / 2024 / Italy",
    )
    small = insert_photo(conn, "bbb", name="b.jpg", md5="dup", size=100)
    conn.close()
    client.post("/scans/exact")
    client.app.state.runner.wait(timeout=10)
    conn = connect(config.db_path)
    gid = conn.execute("SELECT id FROM groups").fetchone()["id"]
    conn.close()
    return gid, big, small


def test_folder_path_links_to_the_drive_folder(client, config, group) -> None:
    pane = client.get("/review/pane?tier=exact")
    assert "https://drive.google.com/drive/folders/folder123" in pane.text
    assert "Photos / 2024 / Italy" in pane.text
    assert 'target="_blank"' in pane.text  # opens in a new tab


def test_each_photo_links_to_its_drive_file(client, config, group) -> None:
    pane = client.get("/review/pane?tier=exact")
    assert "https://drive.google.com/file/d/aaa/view" in pane.text
    assert "https://drive.google.com/file/d/bbb/view" in pane.text


def test_thumbnail_carries_a_lightbox_target(client, config, group) -> None:
    gid, big, _ = group
    pane = client.get("/review/pane?tier=exact")
    assert f'data-full="/photo/{big}/full"' in pane.text


def test_full_route_serves_a_jpeg(client, config, group) -> None:
    gid, big, _ = group
    resp = client.get(f"/photo/{big}/full")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    # the fetcher was asked for the larger lightbox size, not the 512 thumb
    assert ("aaa", 1600) in client.app.state.fetcher.calls


def test_full_route_404_for_missing_photo(client, config, group) -> None:
    resp = client.get("/photo/99999/full")
    assert resp.status_code == 404


def test_lightbox_overlay_is_on_the_full_page(client, config, group) -> None:
    page = client.get("/review?tier=exact")
    assert 'id="lightbox"' in page.text  # the non-blocking overlay element
