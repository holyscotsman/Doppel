import dataclasses
import json

import numpy as np
import pytest

from doppel.config import PerfConfig
from doppel.db import ensure_vec_schema
from doppel.jobs import now
from doppel.stages.adjudicate import run_adjudicate
from tests.fakes import FakeImageFetcher, FakeVlm, insert_photo


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def at_cosine(base: np.ndarray, cosine: float, seed: int = 99) -> np.ndarray:
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


@pytest.fixture
def prompts_dir(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "adjudicate_v1.txt").write_text("adjudicate prompt v1")
    return d


@pytest.fixture
def fetcher(config):
    return FakeImageFetcher(config.cache_dir)


def seed_band_pair(conn, config) -> tuple[int, int]:
    """Two photos with cosine 0.90 — inside [0.85, 0.92)."""
    a = insert_photo(conn, "a", name="a.jpg", md5="ma")
    b = insert_photo(conn, "b", name="b.jpg", md5="mb")
    base = unit(1)
    store_vector(conn, a, base)
    store_vector(conn, b, at_cosine(base, 0.90))
    return a, b


def vlm_groups(conn) -> list[dict]:
    out = []
    for g in conn.execute("SELECT * FROM groups WHERE tier = 'vlm' ORDER BY id"):
        members = {
            row["drive_id"]
            for row in conn.execute(
                """
                SELECT p.drive_id FROM group_members m
                JOIN photos p ON p.id = m.photo_id WHERE m.group_id = ?
                """,
                (g["id"],),
            )
        }
        out.append({"members": members, "color_variant": g["color_variant"]})
    return out


def test_band_pair_gets_stored_verdict_and_group(
    conn, config, fetcher, prompts_dir
) -> None:
    a, b = seed_band_pair(conn, config)
    vlm = FakeVlm([{"verdict": "near", "reason": "same scene, tiny crop"}])

    run_adjudicate(conn, fetcher, vlm, config, prompts_dir=prompts_dir)

    assert len(vlm.calls) == 1
    assert vlm.calls[0]["prompt"] == "adjudicate prompt v1"
    assert vlm.calls[0]["n_images"] == 2

    row = conn.execute("SELECT * FROM vlm_results").fetchone()
    assert (row["photo_id"], row["photo_id_b"]) == (a, b)
    assert row["task"] == "adjudicate"
    assert row["model"] == config.ollama.model
    assert row["prompt_version"] == "v1"
    assert row["verdict"] == "near"
    assert json.loads(row["response"])["reason"] == "same scene, tiny crop"

    assert vlm_groups(conn) == [{"members": {"a", "b"}, "color_variant": 0}]

    scan = conn.execute("SELECT * FROM scans ORDER BY id DESC LIMIT 1").fetchone()
    assert scan["status"] == "done"
    assert scan["processed"] == 1


def test_pairs_outside_band_are_not_adjudicated(
    conn, config, fetcher, prompts_dir
) -> None:
    a = insert_photo(conn, "strong", md5="s")
    b = insert_photo(conn, "weak", md5="w")
    c = insert_photo(conn, "anchor", md5="x")
    base = unit(2)
    store_vector(conn, c, base)
    store_vector(conn, a, at_cosine(base, 0.95))  # above band: similar tier's job
    store_vector(conn, b, at_cosine(base, 0.50, seed=5))  # below band: ignore

    run_adjudicate(conn, fetcher, FakeVlm([]), config, prompts_dir=prompts_dir)

    assert conn.execute("SELECT COUNT(*) AS n FROM vlm_results").fetchone()["n"] == 0


def test_color_variant_near_group_pairs_are_adjudicated(
    conn, config, fetcher, prompts_dir
) -> None:
    a = insert_photo(conn, "red", name="red.jpg", md5="r")
    b = insert_photo(conn, "blue", name="blue.jpg", md5="bl")
    cur = conn.execute(
        "INSERT INTO groups (tier, color_variant, created_at) VALUES ('near', 1, ?)",
        (now(),),
    )
    gid = cur.lastrowid
    conn.executemany(
        "INSERT INTO group_members (group_id, photo_id) VALUES (?, ?)",
        [(gid, a), (gid, b)],
    )
    conn.commit()
    vlm = FakeVlm([{"verdict": "variant", "reason": "same shirt, different color"}])

    run_adjudicate(conn, fetcher, vlm, config, prompts_dir=prompts_dir)

    assert len(vlm.calls) == 1
    assert vlm_groups(conn) == [{"members": {"red", "blue"}, "color_variant": 1}]


def test_different_verdict_creates_no_group(conn, config, fetcher, prompts_dir) -> None:
    seed_band_pair(conn, config)
    vlm = FakeVlm([{"verdict": "different", "reason": "different subjects"}])

    run_adjudicate(conn, fetcher, vlm, config, prompts_dir=prompts_dir)

    assert vlm_groups(conn) == []
    assert conn.execute("SELECT COUNT(*) AS n FROM vlm_results").fetchone()["n"] == 1


def test_resume_skips_already_adjudicated_pairs(
    conn, config, fetcher, prompts_dir
) -> None:
    seed_band_pair(conn, config)
    run_adjudicate(
        conn,
        fetcher,
        FakeVlm([{"verdict": "near", "reason": "x"}]),
        config,
        prompts_dir=prompts_dir,
    )

    # second run: any VLM call would raise AssertionError in FakeVlm
    run_adjudicate(conn, fetcher, FakeVlm([]), config, prompts_dir=prompts_dir)

    assert conn.execute("SELECT COUNT(*) AS n FROM vlm_results").fetchone()["n"] == 1
    scan = conn.execute("SELECT * FROM scans ORDER BY id DESC LIMIT 1").fetchone()
    assert scan["status"] == "done"
    assert scan["total"] == 0


def test_prompt_version_bump_readjudicates_without_clobbering(
    conn, config, fetcher, prompts_dir
) -> None:
    seed_band_pair(conn, config)
    run_adjudicate(
        conn,
        fetcher,
        FakeVlm([{"verdict": "near", "reason": "v1 says near"}]),
        config,
        prompts_dir=prompts_dir,
    )

    (prompts_dir / "adjudicate_v2.txt").write_text("adjudicate prompt v2")
    vlm2 = FakeVlm([{"verdict": "different", "reason": "v2 says different"}])
    run_adjudicate(conn, fetcher, vlm2, config, prompts_dir=prompts_dir)

    assert vlm2.calls[0]["prompt"] == "adjudicate prompt v2"
    rows = conn.execute(
        "SELECT prompt_version, verdict FROM vlm_results ORDER BY id"
    ).fetchall()
    assert [(r["prompt_version"], r["verdict"]) for r in rows] == [
        ("v1", "near"),
        ("v2", "different"),
    ]
    # groups reflect the current prompt version
    assert vlm_groups(conn) == []


def test_interrupt_mid_batch_resumes(conn, config, fetcher, prompts_dir) -> None:
    a, b = seed_band_pair(conn, config)
    c = insert_photo(conn, "c", md5="mc")
    base = unit(1)
    store_vector(conn, c, at_cosine(base, 0.88, seed=42))

    class InterruptingVlm(FakeVlm):
        def chat_json(self, prompt, images, schema):
            if len(self.calls) >= 1:
                raise KeyboardInterrupt
            return super().chat_json(prompt, images, schema)

    vlm = InterruptingVlm([{"verdict": "near", "reason": "first pair"}] * 3)
    # serial so "interrupt on the 2nd pair, exactly 1 stored" is deterministic
    serial = dataclasses.replace(config, perf=PerfConfig(adjudicate_workers=1))
    with pytest.raises(KeyboardInterrupt):
        run_adjudicate(conn, fetcher, vlm, serial, prompts_dir=prompts_dir)

    assert conn.execute("SELECT COUNT(*) AS n FROM vlm_results").fetchone()["n"] == 1
    scan = conn.execute("SELECT * FROM scans ORDER BY id DESC LIMIT 1").fetchone()
    assert scan["status"] == "failed"

    # candidates are (a,b) at 0.90 and (a,c) at 0.88 — pair (b,c) falls
    # below the band. Resume handles only the one remaining pair.
    resumed = FakeVlm([{"verdict": "near", "reason": "second"}])
    run_adjudicate(conn, fetcher, resumed, serial, prompts_dir=prompts_dir)
    assert len(resumed.calls) == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM vlm_results").fetchone()["n"] == 2


def test_parallel_adjudication_rules_every_pair(conn, config, fetcher, prompts_dir):
    """A 3-member color-variant group is 3 pairs; adjudicating them across
    several workers must store all three verdicts and group correctly —
    union-find is order-independent, so concurrency can't change the outcome."""
    ids = [
        insert_photo(conn, name, name=f"{name}.jpg", md5=f"m{name}")
        for name in ("red", "green", "blue")
    ]
    cur = conn.execute(
        "INSERT INTO groups (tier, color_variant, created_at) VALUES ('near', 1, ?)",
        (now(),),
    )
    gid = cur.lastrowid
    conn.executemany(
        "INSERT INTO group_members (group_id, photo_id) VALUES (?, ?)",
        [(gid, pid) for pid in ids],
    )
    conn.commit()

    # every pair comes back "variant"; force real concurrency + a small db_batch
    vlm = FakeVlm([{"verdict": "variant", "reason": "same shot, recolored"}] * 3)
    parallel = dataclasses.replace(
        config, perf=PerfConfig(adjudicate_workers=3, db_batch=2, queue_maxsize=4)
    )
    run_adjudicate(conn, fetcher, vlm, parallel, prompts_dir=prompts_dir)

    assert len(vlm.calls) == 3  # all three pairs adjudicated
    assert conn.execute("SELECT COUNT(*) AS n FROM vlm_results").fetchone()["n"] == 3
    # the three recolors collapse into one vlm group, flagged color_variant
    assert vlm_groups(conn) == [
        {"members": {"red", "green", "blue"}, "color_variant": 1}
    ]
    scan = conn.execute("SELECT * FROM scans ORDER BY id DESC LIMIT 1").fetchone()
    assert scan["status"] == "done" and scan["processed"] == 3
