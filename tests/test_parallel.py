"""The shared concurrency primitives and proof that parallelizing the deeper
scans changes throughput, never results: near/similar produce identical groups
serial (workers=1) and parallel (workers=8), and a failed fetch is retried."""

import dataclasses
import sqlite3
import threading

import numpy as np
import pytest

from doppel.config import PerfConfig
from doppel.db import connect
from doppel.stages.near import run_near
from doppel.stages.parallel import DbWriter, parallel_map
from doppel.stages.similar import run_similar
from tests.fakes import FakeEmbedder, FakeImageFetcher, insert_photo
from tests.images import as_jpeg, structured_image, unrelated_image

# ---- parallel_map -----------------------------------------------------------


def test_parallel_map_returns_every_result_regardless_of_order():
    got = dict(parallel_map(lambda n: n * n, range(20), workers=8))
    assert got == {n: n * n for n in range(20)}


def test_parallel_map_surfaces_exceptions_as_values():
    def half(n):
        if n == 3:
            raise ValueError("boom")
        return n

    results = dict(parallel_map(half, [1, 2, 3, 4], workers=4))
    assert results[1] == 1 and results[4] == 4
    assert isinstance(results[3], ValueError)


def test_parallel_map_workers_one_is_inline_and_ordered():
    seen = list(parallel_map(lambda n: n, [5, 1, 4, 2], workers=1))
    assert [item for item, _ in seen] == [5, 1, 4, 2]


def test_parallel_map_empty():
    assert list(parallel_map(lambda n: n, [], workers=4)) == []


# ---- DbWriter ---------------------------------------------------------------


@pytest.fixture
def scratch(tmp_path):
    conn = connect(tmp_path / "w.db")
    conn.execute("CREATE TABLE t (n INTEGER)")
    conn.commit()
    yield conn
    conn.close()


def test_dbwriter_batches_and_flushes_tail(scratch):
    # 5 rows with db_batch=2 exercises two full batches + a 1-row tail flush
    with DbWriter(scratch, db_batch=2, queue_maxsize=8) as w:
        for n in range(5):
            w.put("INSERT INTO t (n) VALUES (?)", (n,))
    rows = [r["n"] for r in scratch.execute("SELECT n FROM t ORDER BY n")]
    assert rows == [0, 1, 2, 3, 4]


def test_dbwriter_serializes_concurrent_producers(scratch):
    # many threads (like fetch workers) push writes; the single writer must
    # land every row without corruption or loss
    with DbWriter(scratch, db_batch=10, queue_maxsize=16) as w:

        def produce(base):
            for n in range(base, base + 25):
                w.put("INSERT INTO t (n) VALUES (?)", (n,))

        threads = [threading.Thread(target=produce, args=(b,)) for b in (0, 100, 200)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    count = scratch.execute("SELECT COUNT(*) AS c FROM t").fetchone()["c"]
    assert count == 75


def test_dbwriter_reraises_writer_error_on_exit(scratch):
    # a bad write kills the writer thread; the error must surface, not vanish
    with pytest.raises(sqlite3.OperationalError):
        with DbWriter(scratch, db_batch=1, queue_maxsize=4) as w:
            w.put("INSERT INTO does_not_exist (n) VALUES (?)", (1,))


# ---- stage equivalence: serial vs parallel ---------------------------------


def _near_groups(conn):
    groups = []
    for g in conn.execute("SELECT id FROM groups WHERE tier = 'near' ORDER BY id"):
        members = {
            r["drive_id"]
            for r in conn.execute(
                "SELECT p.drive_id FROM group_members m "
                "JOIN photos p ON p.id = m.photo_id WHERE m.group_id = ?",
                (g["id"],),
            )
        }
        groups.append(frozenset(members))
    return sorted(groups, key=sorted)


@pytest.fixture(scope="module")
def near_images():
    base = structured_image()
    return {
        "base": as_jpeg(base),
        "resized": as_jpeg(base.resize((400, 300))),
        "recompressed": as_jpeg(base, quality=25),
        "other": as_jpeg(unrelated_image()),
    }


@pytest.mark.parametrize("workers", [1, 8])
def test_near_groups_identical_serial_and_parallel(
    tmp_path, config, near_images, workers
):
    conn = connect(tmp_path / f"near{workers}.db")
    for name in ("base", "resized", "recompressed", "other"):
        insert_photo(conn, name, name=f"{name}.jpg", md5=f"md5-{name}")
    fetcher = FakeImageFetcher(config.cache_dir / str(workers), images=near_images)
    cfg = dataclasses.replace(
        config, perf=PerfConfig(hash_workers=workers, db_batch=2, queue_maxsize=4)
    )

    run_near(conn, fetcher, cfg)

    # the copies group; unrelated stays out — same with 1 worker or 8
    assert _near_groups(conn) == [frozenset({"base", "resized", "recompressed"})]
    # every fetch happened (set, not order — parallel completes out of order)
    assert ("base", config.thumb_size) in set(fetcher.calls)
    conn.close()


def _similar_groups(conn):
    groups = {}
    for row in conn.execute(
        "SELECT g.id, p.drive_id FROM groups g "
        "JOIN group_members m ON m.group_id = g.id "
        "JOIN photos p ON p.id = m.photo_id WHERE g.tier = 'similar'"
    ):
        groups.setdefault(row["id"], set()).add(row["drive_id"])
    return sorted((frozenset(v) for v in groups.values()), key=sorted)


def _unit(seed):
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def _at_cosine(base, cosine, seed=99):
    noise = _unit(seed)
    orth = noise - np.dot(noise, base) * base
    orth /= np.linalg.norm(orth)
    return (cosine * base + np.sqrt(1 - cosine**2) * orth).astype(np.float32)


@pytest.mark.parametrize("workers", [1, 8])
def test_similar_groups_identical_serial_and_parallel(tmp_path, config, workers):
    conn = connect(tmp_path / f"sim{workers}.db")
    for name in ("a", "b", "c", "d"):
        insert_photo(conn, name, md5=f"m{name}")
    base = _unit(1)
    embedder = FakeEmbedder(
        {
            "a": base,
            "b": _at_cosine(base, 0.97),
            "c": _at_cosine(base, 0.96, seed=5),
            "d": _at_cosine(base, 0.4, seed=7),
        }
    )
    fetcher = FakeImageFetcher(config.cache_dir / f"s{workers}")
    cfg = dataclasses.replace(
        config,
        perf=PerfConfig(embed_fetch_workers=workers, clip_batch=2, db_batch=2),
    )

    run_similar(conn, fetcher, embedder, cfg)

    # a,b,c cluster (all mutually >= threshold via a); d stays out — invariant
    # to worker count and CLIP batch composition
    assert _similar_groups(conn) == [frozenset({"a", "b", "c"})]
    # all four embedded exactly once, in whatever batch order
    assert sorted(sum(embedder.calls, [])) == ["a", "b", "c", "d"]
    conn.close()


# ---- resumability under a failed fetch -------------------------------------


def test_near_failed_fetch_is_left_unhashed_and_retried(tmp_path, config, near_images):
    conn = connect(tmp_path / "retry.db")
    for name in ("base", "resized"):
        insert_photo(conn, name, name=f"{name}.jpg", md5=f"md5-{name}")
    insert_photo(conn, "flaky", name="flaky.jpg", md5="md5-flaky")

    # 'flaky' raises on first pass; the others hash normally
    imgs = dict(near_images)
    imgs["flaky"] = RuntimeError("transient fetch failure")
    cfg = dataclasses.replace(config, perf=PerfConfig(hash_workers=8, db_batch=2))
    run_near(conn, FakeImageFetcher(config.cache_dir / "r1", images=imgs), cfg)

    hashed = {
        r["drive_id"]: r["phash"]
        for r in conn.execute("SELECT drive_id, phash FROM photos")
    }
    assert hashed["base"] is not None and hashed["resized"] is not None
    assert hashed["flaky"] is None  # failure left it un-hashed, not wedged

    # second run with a healthy fetch picks up exactly the un-hashed photo
    good = dict(near_images)
    good["flaky"] = as_jpeg(structured_image())
    run_near(conn, FakeImageFetcher(config.cache_dir / "r2", images=good), cfg)
    again = conn.execute("SELECT phash FROM photos WHERE drive_id = 'flaky'").fetchone()
    assert again["phash"] is not None
    conn.close()
