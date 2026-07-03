"""Stage 3 — similar: same scene/subject, different shot. CLIP embeddings
stored in sqlite-vec; kNN pairing; union-find clusters.

Grouping reads stored vectors, so threshold changes never require
re-embedding (SPEC acceptance for Phase 4).
"""

from __future__ import annotations

import sqlite3
import traceback
from pathlib import Path

import numpy as np

from doppel.config import Config
from doppel.db import ensure_vec_schema
from doppel.drive import ImageFetcher
from doppel.embed import BATCH_SIZE, Embedder
from doppel.jobs import fail_scan, finish_scan, now, start_scan
from doppel.stages.exact import rebuild_groups
from doppel.stages.grouping import UnionFind

K_NEIGHBORS = 20  # per SPEC "Stage 3 — similar"


def _embed_missing(
    conn: sqlite3.Connection,
    fetcher: ImageFetcher,
    embedder: Embedder,
    config: Config,
    scan_id: int,
) -> int:
    """Embed every active photo without a stored vector. Resumable: photos
    embedded on a previous run are excluded by the query."""
    todo = conn.execute(
        """
        SELECT id, drive_id FROM photos
        WHERE status = 'active'
          AND id NOT IN (SELECT photo_id FROM embeddings)
        ORDER BY id
        """
    ).fetchall()
    conn.execute("UPDATE scans SET total = ? WHERE id = ?", (len(todo), scan_id))
    conn.commit()
    processed = 0
    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start : start + BATCH_SIZE]
        rows: list[sqlite3.Row] = []
        paths: list[Path] = []
        for row in batch:
            try:
                paths.append(fetcher.get(row["drive_id"], config.thumb_size))
            except Exception:
                # leave un-embedded: the next run retries this photo
                traceback.print_exc()
            else:
                rows.append(row)
        if rows:
            vectors = embedder.embed(paths)
            for row, vector in zip(rows, vectors, strict=True):
                conn.execute(
                    "INSERT INTO embeddings (photo_id, embedding) VALUES (?, ?)",
                    (row["id"], vector.astype(np.float32).tobytes()),
                )
        processed += len(batch)
        conn.execute(
            "UPDATE scans SET processed = ? WHERE id = ?", (processed, scan_id)
        )
        conn.commit()
    return len(todo)


def load_vectors(conn: sqlite3.Connection) -> dict[int, np.ndarray]:
    """Stored vectors for active photos only."""
    return {
        row["photo_id"]: np.frombuffer(row["embedding"], dtype=np.float32)
        for row in conn.execute(
            """
            SELECT e.photo_id, e.embedding FROM embeddings e
            JOIN photos p ON p.id = e.photo_id
            WHERE p.status = 'active'
            """
        )
    }


def _similar_pairs(
    conn: sqlite3.Connection,
    vectors: dict[int, np.ndarray],
    config: Config,
) -> set[tuple[int, int]]:
    """kNN per photo via sqlite-vec; keep pairs at cosine >= the threshold.

    kNN is asymmetric under the k cap: when A sits in a dense burst, B can
    be crowded out of A's top-k while A is still in B's. Pairs are kept
    from whichever direction finds them, normalized to (low, high) id.
    """
    pairs: set[tuple[int, int]] = set()
    for photo_id, vector in vectors.items():
        neighbors = conn.execute(
            "SELECT photo_id, distance FROM embeddings "
            "WHERE embedding MATCH ? AND k = ?",
            (vector.tobytes(), K_NEIGHBORS),
        ).fetchall()
        for row in neighbors:
            other = row["photo_id"]
            cosine = 1.0 - row["distance"]
            if other == photo_id or other not in vectors:
                continue  # self or inactive photo
            if cosine >= config.similar_cosine_min:
                pairs.add((min(photo_id, other), max(photo_id, other)))
    return pairs


def _existing_group_member_sets(conn: sqlite3.Connection) -> list[set[int]]:
    sets: dict[int, set[int]] = {}
    for row in conn.execute(
        """
        SELECT g.id, m.photo_id FROM groups g
        JOIN group_members m ON m.group_id = g.id
        WHERE g.tier IN ('exact', 'near')
        """
    ):
        sets.setdefault(row["id"], set()).add(row["photo_id"])
    return list(sets.values())


def _ensure_embedding_space(conn: sqlite3.Connection, config: Config) -> None:
    """Vectors from different CLIP models live in unrelated spaces; mixing
    them makes cosine meaningless. On a model change, wipe stored vectors so
    everything re-embeds under the new model."""
    row = conn.execute("SELECT value FROM meta WHERE key = 'clip_model'").fetchone()
    if row is not None and row["value"] == config.clip_model:
        return
    if row is not None:
        conn.execute("DELETE FROM embeddings")
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('clip_model', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (config.clip_model,),
    )
    conn.commit()


def run_similar(
    conn: sqlite3.Connection,
    fetcher: ImageFetcher,
    embedder: Embedder,
    config: Config,
) -> int:
    """Embed, pair, cluster; drop clusters adding nothing over exact/near."""
    scan_id = start_scan(conn, "similar")
    try:
        ensure_vec_schema(conn)
        _ensure_embedding_space(conn, config)
        n_embedded = _embed_missing(conn, fetcher, embedder, config, scan_id)

        vectors = load_vectors(conn)
        uf = UnionFind()
        for a, b in _similar_pairs(conn, vectors, config):
            uf.union(a, b)

        existing = _existing_group_member_sets(conn)
        clusters = [
            cluster
            for cluster in uf.clusters()
            if not any(set(cluster) <= members for members in existing)
        ]

        rebuild_groups(conn, "similar")
        for cluster in clusters:
            anchor = vectors[cluster[0]]
            cur = conn.execute(
                "INSERT INTO groups (tier, created_at) VALUES ('similar', ?)",
                (now(),),
            )
            group_id = int(cur.lastrowid)
            for photo_id in cluster:
                score = float(np.dot(vectors[photo_id], anchor))
                conn.execute(
                    "INSERT INTO group_members (group_id, photo_id, score) "
                    "VALUES (?, ?, ?)",
                    (group_id, photo_id, score),
                )
        conn.commit()
        finish_scan(conn, scan_id, total=n_embedded)
    except BaseException as exc:
        conn.rollback()
        fail_scan(conn, scan_id, f"{type(exc).__name__}: {exc}")
        raise
    return scan_id
