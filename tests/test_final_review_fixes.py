"""Regression tests for the confirmed final-review findings."""

import sqlite3

import numpy as np
from fastapi.testclient import TestClient

from doppel.app import create_app
from doppel.db import connect, ensure_vec_schema
from doppel.jobs import run_sync
from doppel.stages.adjudicate import run_adjudicate
from doppel.stages.near import run_near
from doppel.stages.similar import run_similar
from tests.fakes import (
    FakeDriveClient,
    FakeEmbedder,
    FakeImageFetcher,
    FakeVlm,
    insert_photo,
    make_file,
)
from tests.images import as_jpeg, structured_image


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def at_cosine(base: np.ndarray, cosine: float, seed: int) -> np.ndarray:
    noise = unit(seed)
    orth = noise - np.dot(noise, base) * base
    orth /= np.linalg.norm(orth)
    return (cosine * base + np.sqrt(1 - cosine**2) * orth).astype(np.float32)


def store_vector(conn, photo_id: int, vector: np.ndarray) -> None:
    ensure_vec_schema(conn)
    conn.execute(
        "INSERT INTO embeddings (photo_id, embedding) VALUES (?, ?)",
        (photo_id, vector.astype(np.float32).tobytes()),
    )
    conn.commit()


def seed_burst_clique(conn, n: int = 21) -> np.ndarray:
    """A burst of n nearly identical shots that saturates any k=20 list."""
    base = unit(1)
    for i in range(n):
        pid = insert_photo(conn, f"burst{i}", md5=f"mb{i}")
        store_vector(conn, pid, at_cosine(base, 0.999, seed=1000 + i))
    return base


def test_knn_asymmetry_does_not_drop_similar_pairs(conn, config) -> None:
    """A photo crowded out of a burst clique's k=20 lists must still group
    when its own kNN query finds above-threshold pairs."""
    base = seed_burst_clique(conn)
    outsider = insert_photo(conn, "outsider", md5="mo")
    store_vector(conn, outsider, at_cosine(base, 0.93, seed=7))

    run_similar(conn, FakeImageFetcher(config.cache_dir), FakeEmbedder({}), config)

    grouped = {
        row["photo_id"]
        for row in conn.execute(
            """
            SELECT m.photo_id FROM group_members m
            JOIN groups g ON g.id = m.group_id WHERE g.tier = 'similar'
            """
        )
    }
    assert outsider in grouped


def test_knn_asymmetry_does_not_drop_band_pairs(conn, config, tmp_path) -> None:
    """Borderline pairs crowded out of the clique's k=20 lists must still
    reach the VLM via the outsider's own kNN query."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "adjudicate_v1.txt").write_text("p1")

    base = seed_burst_clique(conn)
    outsider = insert_photo(conn, "outsider", md5="mo")
    store_vector(conn, outsider, at_cosine(base, 0.88, seed=7))

    vlm = FakeVlm([{"verdict": "near", "reason": "x"}] * 30)
    run_adjudicate(
        conn, FakeImageFetcher(config.cache_dir), vlm, config, prompts_dir=prompts
    )

    adjudicated = {
        row["photo_id_b"] for row in conn.execute("SELECT photo_id_b FROM vlm_results")
    } | {row["photo_id"] for row in conn.execute("SELECT photo_id FROM vlm_results")}
    assert outsider in adjudicated


def test_inplace_edit_invalidates_derived_state(conn, config, tmp_path) -> None:
    """Same drive_id, new md5 -> hashes, embedding, and cache are dropped."""
    cache = tmp_path / "cache"
    cache.mkdir()
    run_sync(conn, FakeDriveClient([make_file("x", md5="old")]), cache)
    pid = conn.execute("SELECT id FROM photos WHERE drive_id = 'x'").fetchone()["id"]
    conn.execute(
        "UPDATE photos SET phash = 'aa', dhash = 'bb', thumb_path = 'p' WHERE id = ?",
        (pid,),
    )
    store_vector(conn, pid, unit(3))
    (cache / "x_512.jpg").write_bytes(b"old-pixels")
    (cache / "x_orig").write_bytes(b"old-orig")

    run_sync(conn, FakeDriveClient([make_file("x", md5="new")]), cache)

    row = conn.execute("SELECT * FROM photos WHERE id = ?", (pid,)).fetchone()
    assert row["phash"] is None and row["dhash"] is None
    assert row["thumb_path"] is None
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM embeddings WHERE photo_id = ?", (pid,)
    ).fetchone()["n"]
    assert n == 0
    assert not (cache / "x_512.jpg").exists()
    assert not (cache / "x_orig").exists()


def test_unchanged_file_keeps_derived_state(conn, config, tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    run_sync(conn, FakeDriveClient([make_file("x", md5="same")]), cache)
    pid = conn.execute("SELECT id FROM photos WHERE drive_id = 'x'").fetchone()["id"]
    conn.execute("UPDATE photos SET phash = 'aa', dhash = 'bb' WHERE id = ?", (pid,))
    conn.commit()

    run_sync(conn, FakeDriveClient([make_file("x", md5="same")]), cache)

    row = conn.execute("SELECT phash FROM photos WHERE id = ?", (pid,)).fetchone()
    assert row["phash"] == "aa"


def test_clip_model_change_wipes_and_reembeds(conn, config) -> None:
    import dataclasses

    insert_photo(conn, "a", md5="ma")
    insert_photo(conn, "b", md5="mb")
    base = unit(1)
    fetcher = FakeImageFetcher(config.cache_dir)
    first = FakeEmbedder({"a": base, "b": at_cosine(base, 0.95, seed=2)})
    run_similar(conn, fetcher, first, config)
    assert sorted(sum(first.calls, [])) == ["a", "b"]

    switched = dataclasses.replace(config, clip_model="ViT-B-16/laion2b_s34b_b88k")
    second = FakeEmbedder({"a": base, "b": at_cosine(base, 0.95, seed=2)})
    run_similar(conn, fetcher, second, config=switched)

    # everything re-embedded under the new model, not mixed
    assert sorted(sum(second.calls, [])) == ["a", "b"]

    # same model again: nothing re-embeds
    third = FakeEmbedder({})
    run_similar(conn, fetcher, third, config=switched)
    assert third.calls == []


class DbWritingFetcher(FakeImageFetcher):
    """Mimics DriveImageFetcher: records thumb_path on a SECOND connection
    with a short busy timeout — deadlocks if called inside an open write
    transaction on the main connection."""

    def __init__(self, cache_dir, db_path, images=None) -> None:
        super().__init__(cache_dir, images)
        self.db_path = db_path

    def get(self, drive_id: str, size=512):
        path = super().get(drive_id, size)
        side = sqlite3.connect(self.db_path, timeout=0.5)
        try:
            side.execute(
                "UPDATE photos SET thumb_path = ? WHERE drive_id = ?",
                (str(path), drive_id),
            )
            side.commit()
        finally:
            side.close()
        return path


def test_near_grouping_does_not_deadlock_dbwriting_fetcher(config) -> None:
    """Colorway flagging fetches thumbnails; the real fetcher writes
    thumb_path on its own connection. That must not run inside the grouping
    write transaction."""
    conn = connect(config.db_path)
    base = structured_image()
    images = {"n1": as_jpeg(base), "n2": as_jpeg(base.resize((400, 300)))}
    insert_photo(conn, "n1", md5="m1")
    insert_photo(conn, "n2", md5="m2")
    fetcher = DbWritingFetcher(config.cache_dir, config.db_path, images=images)

    run_near(conn, fetcher, config)  # raises sqlite3.OperationalError on regression

    n = conn.execute("SELECT COUNT(*) AS n FROM groups WHERE tier = 'near'").fetchone()[
        "n"
    ]
    assert n == 1
    thumb = conn.execute(
        "SELECT thumb_path FROM photos WHERE drive_id = 'n1'"
    ).fetchone()["thumb_path"]
    assert thumb is not None


def test_contrast_edit_is_not_flagged_as_colorway(conn, config) -> None:
    from PIL import ImageEnhance

    base = structured_image()
    contrast = ImageEnhance.Contrast(base).enhance(1.5)
    images = {"orig": as_jpeg(base), "contrast": as_jpeg(contrast)}
    insert_photo(conn, "orig", md5="a")
    insert_photo(conn, "contrast", md5="b")

    run_near(conn, FakeImageFetcher(config.cache_dir, images=images), config)

    group = conn.execute("SELECT * FROM groups WHERE tier = 'near'").fetchone()
    assert group is not None  # contrast edit still groups as a near-dup
    assert group["color_variant"] == 0  # ...but is not a colorway


def test_stage_dependency_failure_lands_in_ledger(config) -> None:
    from doppel.drive import CredentialsRequired

    def failing_factory(cfg):
        raise CredentialsRequired("Drive authorization required — run make scan")

    app = create_app(config=config, fetcher_factory=failing_factory)
    with TestClient(app) as ui:
        conn = connect(config.db_path)
        insert_photo(conn, "p", md5="m")
        conn.close()

        resp = ui.post("/scans/near")
        assert resp.status_code == 200
        app.state.runner.wait(timeout=10)

        conn = connect(config.db_path)
        scan = conn.execute(
            "SELECT * FROM scans WHERE stage = 'near' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        assert scan is not None
        assert scan["status"] == "failed"
        assert "authorization required" in scan["error"].lower()


def test_startup_reconciles_orphaned_running_scans(config) -> None:
    conn = connect(config.db_path)
    conn.execute(
        "INSERT INTO scans (stage, status, processed, started_at) "
        "VALUES ('near', 'running', 7, 'x')"
    )
    conn.commit()
    conn.close()

    app = create_app(
        config=config, fetcher_factory=lambda cfg: FakeImageFetcher(cfg.cache_dir)
    )
    with TestClient(app):
        pass

    conn = connect(config.db_path)
    row = conn.execute("SELECT status, error FROM scans").fetchone()
    conn.close()
    assert row["status"] == "failed"
    assert row["error"] == "interrupted"


def test_decisions_ignore_malformed_field_names(config) -> None:
    app = create_app(
        config=config, fetcher_factory=lambda cfg: FakeImageFetcher(cfg.cache_dir)
    )
    with TestClient(app) as ui:
        conn = connect(config.db_path)
        a = insert_photo(conn, "a", md5="d", size=200)
        insert_photo(conn, "b", md5="d", size=100)
        conn.close()
        ui.post("/scans/exact")
        app.state.runner.wait(timeout=10)
        conn = connect(config.db_path)
        gid = conn.execute("SELECT id FROM groups").fetchone()["id"]
        conn.close()

        resp = ui.post(
            f"/groups/{gid}/decisions",
            data={f"action_{a}": "keep", "action_abc": "trash", "action_": "keep"},
            follow_redirects=False,
        )

        assert resp.status_code == 303  # not a 500
        conn = connect(config.db_path)
        row = conn.execute(
            "SELECT action FROM decisions WHERE photo_id = ?", (a,)
        ).fetchone()
        conn.close()
        assert row["action"] == "keep"  # the valid field was saved
