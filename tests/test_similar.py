import dataclasses

import numpy as np
import pytest

from doppel.stages.exact import run_exact
from doppel.stages.similar import run_similar
from tests.fakes import FakeEmbedder, FakeImageFetcher, insert_photo


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def at_cosine(base: np.ndarray, cosine: float, seed: int = 99) -> np.ndarray:
    """A unit vector at exactly the given cosine similarity to base."""
    noise = unit(seed)
    orth = noise - np.dot(noise, base) * base
    orth /= np.linalg.norm(orth)
    return (cosine * base + np.sqrt(1 - cosine**2) * orth).astype(np.float32)


@pytest.fixture
def fetcher(config):
    return FakeImageFetcher(config.cache_dir)


def similar_groups(conn) -> list[set[str]]:
    groups: dict[int, set[str]] = {}
    for row in conn.execute(
        """
        SELECT g.id, p.drive_id FROM groups g
        JOIN group_members m ON m.group_id = g.id
        JOIN photos p ON p.id = m.photo_id
        WHERE g.tier = 'similar'
        """
    ):
        groups.setdefault(row["id"], set()).add(row["drive_id"])
    return sorted(groups.values(), key=sorted)


def test_similar_clusters_at_threshold(conn, config, fetcher) -> None:
    base = unit(1)
    insert_photo(conn, "a", md5="ma")
    insert_photo(conn, "b", md5="mb")
    insert_photo(conn, "c", md5="mc")
    embedder = FakeEmbedder(
        {"a": base, "b": at_cosine(base, 0.95), "c": at_cosine(base, 0.5, seed=7)}
    )

    run_similar(conn, fetcher, embedder, config)

    assert similar_groups(conn) == [{"a", "b"}]


def test_similar_scores_are_cosine_vs_anchor(conn, config, fetcher) -> None:
    base = unit(1)
    insert_photo(conn, "a", md5="ma")
    insert_photo(conn, "b", md5="mb")
    embedder = FakeEmbedder({"a": base, "b": at_cosine(base, 0.95)})

    run_similar(conn, fetcher, embedder, config)

    scores = {
        row["drive_id"]: row["score"]
        for row in conn.execute(
            """
            SELECT p.drive_id, m.score FROM group_members m
            JOIN photos p ON p.id = m.photo_id
            JOIN groups g ON g.id = m.group_id WHERE g.tier = 'similar'
            """
        )
    }
    assert scores["a"] == pytest.approx(1.0, abs=1e-5)
    assert scores["b"] == pytest.approx(0.95, abs=1e-3)


def test_threshold_change_regroups_without_reembedding(conn, config, fetcher) -> None:
    base = unit(1)
    insert_photo(conn, "a", md5="ma")
    insert_photo(conn, "b", md5="mb")
    run_similar(
        conn, fetcher, FakeEmbedder({"a": base, "b": at_cosine(base, 0.95)}), config
    )
    assert similar_groups(conn) == [{"a", "b"}]

    strict = dataclasses.replace(config, similar_cosine_min=0.99)
    strict_embedder = FakeEmbedder({})  # any embed call would raise KeyError

    run_similar(conn, fetcher, strict_embedder, strict)

    assert strict_embedder.calls == []
    assert similar_groups(conn) == []


def test_similar_embedding_resumes_incrementally(conn, config, fetcher) -> None:
    base = unit(1)
    insert_photo(conn, "a", md5="ma")
    insert_photo(conn, "b", md5="mb")
    first = FakeEmbedder({"a": base, "b": at_cosine(base, 0.95)})
    run_similar(conn, fetcher, first, config)
    assert sorted(sum(first.calls, [])) == ["a", "b"]

    insert_photo(conn, "c", md5="mc")
    second = FakeEmbedder({"c": at_cosine(base, 0.5, seed=7)})

    run_similar(conn, fetcher, second, config)

    assert sum(second.calls, []) == ["c"]  # a and b were not re-embedded


def test_cluster_contained_in_exact_group_is_dropped(conn, config, fetcher) -> None:
    base = unit(1)
    insert_photo(conn, "a", md5="same")
    insert_photo(conn, "b", md5="same")  # byte-identical pair
    run_exact(conn)
    embedder = FakeEmbedder({"a": base, "b": at_cosine(base, 0.99)})

    run_similar(conn, fetcher, embedder, config)

    assert similar_groups(conn) == []


def test_cluster_exceeding_exact_group_is_kept(conn, config, fetcher) -> None:
    base = unit(1)
    insert_photo(conn, "a", md5="same")
    insert_photo(conn, "b", md5="same")
    insert_photo(conn, "c", md5="other")  # related shot, not byte-identical
    run_exact(conn)
    embedder = FakeEmbedder(
        {"a": base, "b": at_cosine(base, 0.99), "c": at_cosine(base, 0.94, seed=3)}
    )

    run_similar(conn, fetcher, embedder, config)

    assert similar_groups(conn) == [{"a", "b", "c"}]


def test_similar_ignores_missing_photos(conn, config, fetcher) -> None:
    base = unit(1)
    insert_photo(conn, "a", md5="ma")
    insert_photo(conn, "b", md5="mb", status="missing")
    embedder = FakeEmbedder({"a": base})

    run_similar(conn, fetcher, embedder, config)

    assert sum(embedder.calls, []) == ["a"]
    assert similar_groups(conn) == []


def test_similar_rebuild_is_idempotent(conn, config, fetcher) -> None:
    base = unit(1)
    insert_photo(conn, "a", md5="ma")
    insert_photo(conn, "b", md5="mb")
    embedder = FakeEmbedder({"a": base, "b": at_cosine(base, 0.95)})

    run_similar(conn, fetcher, embedder, config)
    run_similar(conn, fetcher, FakeEmbedder({}), config)

    assert similar_groups(conn) == [{"a", "b"}]


def test_similar_failure_marks_scan_failed(conn, config, fetcher) -> None:
    insert_photo(conn, "a", md5="ma")

    class Boom(FakeEmbedder):
        def embed(self, paths):
            raise RuntimeError("model exploded")

    with pytest.raises(RuntimeError):
        run_similar(conn, fetcher, Boom({}), config)

    scan = conn.execute("SELECT * FROM scans ORDER BY id DESC LIMIT 1").fetchone()
    assert scan["status"] == "failed"
    assert "model exploded" in scan["error"]
