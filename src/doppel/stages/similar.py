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
from PIL import Image

from doppel.config import Config
from doppel.db import ensure_vec_schema
from doppel.drive import ImageFetcher
from doppel.embed import BATCH_SIZE, Embedder
from doppel.jobs import (
    fail_scan,
    finish_scan,
    now,
    reprocess_tail,
    stage_was_interrupted,
    start_scan,
)
from doppel.stages.exact import rebuild_groups
from doppel.stages.grouping import UnionFind
from doppel.stages.parallel import DbWriter, parallel_map

K_NEIGHBORS = 20  # per SPEC "Stage 3 — similar"


def _embed_missing(
    conn: sqlite3.Connection,
    fetcher: ImageFetcher,
    embedder: Embedder,
    config: Config,
    scan_id: int,
) -> int:
    """Embed every active photo without a stored vector. Resumable: photos
    embedded on a previous run are excluded by the query.

    Three roles run concurrently: ``perf.embed_fetch_workers`` threads download
    thumbnails, the single main thread batches them ``perf.clip_batch`` at a
    time through the embedder (CLIP is called from ONE thread only — MPS is not
    thread-safe), and a DbWriter thread owns every INSERT. A photo whose fetch
    fails is left un-embedded and retried next run. Fetch order doesn't affect
    correctness: each vector is stored against its own photo id, and grouping
    reads them back independently.
    """
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

    perf = config.perf
    clip_batch = max(1, perf.clip_batch or BATCH_SIZE)

    def fetch_one(row: sqlite3.Row) -> Path:
        path = fetcher.get(row["drive_id"], config.thumb_size)
        # embedder.embed() decodes on the MAIN thread, outside parallel_map's
        # per-item guard, so one undecodable file there would crash the whole
        # stage. Decode here in the worker instead: a bad file raises (and is
        # yielded as a skippable per-item error), and we drop the poisoned cache
        # entry so the next run re-fetches it fresh through the validated path.
        try:
            with Image.open(path) as img:
                img.load()
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path

    processed = 0
    buf_rows: list[sqlite3.Row] = []
    buf_paths: list[Path] = []

    with DbWriter(conn, perf.db_batch, perf.queue_maxsize) as writer:

        def flush_embeddings() -> None:
            if not buf_rows:
                return
            vectors = embedder.embed(buf_paths)  # main thread only (MPS)
            for r, vector in zip(buf_rows, vectors, strict=True):
                writer.put(
                    "INSERT INTO embeddings (photo_id, embedding) VALUES (?, ?)",
                    (r["id"], vector.astype(np.float32).tobytes()),
                )
            buf_rows.clear()
            buf_paths.clear()

        for row, result in parallel_map(fetch_one, todo, perf.embed_fetch_workers):
            processed += 1
            if isinstance(result, Exception):
                traceback.print_exception(result)  # retried next run
            else:
                buf_rows.append(row)
                buf_paths.append(result)
                if len(buf_rows) >= clip_batch:
                    flush_embeddings()
            writer.put(
                "UPDATE scans SET processed = ? WHERE id = ?", (processed, scan_id)
            )
        flush_embeddings()  # tail
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
    resuming = stage_was_interrupted(conn, "similar")
    scan_id = start_scan(conn, "similar")
    try:
        ensure_vec_schema(conn)
        _ensure_embedding_space(conn, config)
        if resuming:
            # re-embed the last few from a fresh fetch (crash-boundary safety)
            reprocess_tail(conn, "similar", config.resume_overlap, config.cache_dir)
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
