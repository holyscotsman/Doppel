"""Interrupt-safe resume: progress persists, and on resume a stage re-does the
last few finished items (config.resume_overlap) in case the crash corrupted the
work right at that boundary."""

import dataclasses

from doppel.db import connect, ensure_vec_schema
from doppel.jobs import now, reprocess_tail, stage_was_interrupted
from doppel.stages.near import run_near
from tests.fakes import FakeImageFetcher, insert_photo
from tests.images import as_jpeg, structured_image, unrelated_image


def _mark_scan(conn, stage, status, error=None):
    conn.execute(
        "INSERT INTO scans (stage, status, error, processed, started_at) "
        "VALUES (?, ?, ?, 0, ?)",
        (stage, status, error, now()),
    )
    conn.commit()


def test_stage_was_interrupted_only_on_the_interrupted_sentinel(conn):
    assert stage_was_interrupted(conn, "near") is False  # never run
    _mark_scan(conn, "near", "done")
    assert stage_was_interrupted(conn, "near") is False
    # a genuine failure (real error string) is NOT a resume — just retry it
    _mark_scan(conn, "near", "failed", error="RuntimeError: drive 500")
    assert stage_was_interrupted(conn, "near") is False
    # a crash/kill is reconciled to error='interrupted' — that IS a resume
    _mark_scan(conn, "near", "failed", error="interrupted")
    assert stage_was_interrupted(conn, "near") is True
    _mark_scan(conn, "near", "done")  # a later clean run clears it
    assert stage_was_interrupted(conn, "near") is False


def test_reprocess_tail_near_clears_last_n_hashes(conn):
    ids = [insert_photo(conn, f"p{i}", md5=f"m{i}") for i in range(5)]
    conn.execute("UPDATE photos SET phash = 'h', dhash = 'd'")
    conn.commit()

    cleared = reprocess_tail(conn, "near", 2)

    assert cleared == [ids[4], ids[3]]  # highest ids (most recently finished)
    state = {r["id"]: r["phash"] for r in conn.execute("SELECT id, phash FROM photos")}
    assert state[ids[0]] == "h" and state[ids[1]] == "h" and state[ids[2]] == "h"
    assert state[ids[3]] is None and state[ids[4]] is None  # queued for redo


def test_reprocess_tail_similar_deletes_last_n_embeddings(conn):
    import numpy as np

    ensure_vec_schema(conn)
    ids = [insert_photo(conn, f"p{i}", md5=f"m{i}") for i in range(4)]
    for pid in ids:
        conn.execute(
            "INSERT INTO embeddings (photo_id, embedding) VALUES (?, ?)",
            (pid, np.zeros(512, dtype=np.float32).tobytes()),
        )
    conn.commit()

    reprocess_tail(conn, "similar", 1)

    remaining = {r["photo_id"] for r in conn.execute("SELECT photo_id FROM embeddings")}
    assert remaining == set(ids[:3])  # the highest id was dropped for re-embed


def test_reprocess_tail_adjudicate_is_a_noop(conn):
    # adjudicate is intentionally excluded — a resumed adjudicate re-asks pairs
    # that lack a verdict; deleting one whose pair is no longer a candidate
    # would silently lose it, so reprocess_tail must never touch vlm_results.
    a = insert_photo(conn, "a", md5="a")
    b = insert_photo(conn, "b", md5="b")
    conn.execute(
        "INSERT INTO vlm_results (task, photo_id, photo_id_b, model, "
        "prompt_version, response, verdict, created_at) "
        "VALUES ('adjudicate', ?, ?, 'm', 'v1', '{}', 'near', ?)",
        (a, b, now()),
    )
    conn.commit()

    assert reprocess_tail(conn, "adjudicate", 5) == []
    left = conn.execute("SELECT COUNT(*) AS n FROM vlm_results").fetchone()["n"]
    assert left == 1  # untouched


def test_reprocess_tail_zero_is_a_noop(conn):
    insert_photo(conn, "p", md5="m")
    conn.execute("UPDATE photos SET phash = 'h', dhash = 'd'")
    conn.commit()
    assert reprocess_tail(conn, "near", 0) == []
    assert conn.execute("SELECT phash FROM photos").fetchone()["phash"] == "h"


def test_near_resume_redoes_only_the_overlap(config):
    images = {
        "base": as_jpeg(structured_image()),
        "resized": as_jpeg(structured_image().resize((400, 300))),
        "recompressed": as_jpeg(structured_image(), quality=25),
        "other": as_jpeg(unrelated_image()),
    }
    conn = connect(config.db_path)
    for name in ("base", "resized", "recompressed", "other"):
        insert_photo(conn, name, name=f"{name}.jpg", md5=f"md5-{name}")
    cfg = dataclasses.replace(config, resume_overlap=2)

    # clean run hashes everything
    run_near(conn, FakeImageFetcher(config.cache_dir / "a", images=images), cfg)
    assert all(r["phash"] for r in conn.execute("SELECT phash FROM photos"))

    # simulate a crash: the last near scan is left interrupted
    conn.execute(
        "UPDATE scans SET status = 'failed', error = 'interrupted' WHERE stage = 'near'"
    )
    conn.commit()

    # resume: exactly resume_overlap photos are re-hashed, not the whole library
    run_near(conn, FakeImageFetcher(config.cache_dir / "b", images=images), cfg)
    resume_scan = conn.execute(
        "SELECT total FROM scans WHERE stage = 'near' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert resume_scan["total"] == 2
    # and everything ends up hashed again
    assert all(r["phash"] for r in conn.execute("SELECT phash FROM photos"))
    conn.close()
