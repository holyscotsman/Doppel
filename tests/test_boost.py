"""Phase 3 (#6): the Boost toggle and the boosted perf it applies to scans."""

from __future__ import annotations

import dataclasses

import pytest
from fastapi.testclient import TestClient

from doppel.app import BOOST_PERF, create_app
from doppel.config import PerfConfig
from doppel.db import connect, get_meta
from tests.fakes import FakeImageFetcher


@pytest.fixture
def client(config):
    app = create_app(
        config=config, fetcher_factory=lambda cfg: FakeImageFetcher(cfg.cache_dir)
    )
    with TestClient(app) as c:
        yield c


def _boost(config) -> str | None:
    conn = connect(config.db_path)
    try:
        return get_meta(conn, "boost", "off")
    finally:
        conn.close()


def test_boost_defaults_off_and_toggles(client, config) -> None:
    assert _boost(config) == "off"
    client.post("/boost", follow_redirects=False)
    assert _boost(config) == "on"
    client.post("/boost", follow_redirects=False)
    assert _boost(config) == "off"


def test_boost_perf_keys_are_valid_perfconfig_fields() -> None:
    fields = {f.name for f in dataclasses.fields(PerfConfig)}
    assert set(BOOST_PERF) <= fields  # dataclasses.replace(**BOOST_PERF) won't raise


def test_boost_perf_actually_boosts_over_the_defaults() -> None:
    base = PerfConfig()
    boosted = dataclasses.replace(base, **BOOST_PERF)
    # the stages read hash_workers (near) and embed_fetch_workers (similar)
    assert boosted.hash_workers == 64 and boosted.clip_batch == 96
    assert boosted.embed_fetch_workers > base.embed_fetch_workers
    assert boosted.queue_maxsize >= base.queue_maxsize
