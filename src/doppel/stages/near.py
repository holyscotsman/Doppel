"""Stage 2 — near duplicates: the same image re-encoded, resized, or lightly
edited. Perceptual hashes (pHash + dHash) over 512px thumbnails.

Hashes are computed on grayscale (imagehash converts internally), so
desaturated or re-tinted copies land here by design — and so do different
colorways of one base shot. Groups whose members differ in HSV color
distribution are flagged color_variant for the UI badge and Phase 6
adjudication.
"""

from __future__ import annotations

import sqlite3
import traceback
from itertools import combinations
from pathlib import Path

import imagehash
import numpy as np
import pybktree
from PIL import Image

from doppel.config import Config
from doppel.drive import ImageFetcher
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

# structural constants of the colorway metric (not tunable thresholds):
# coarse bins tolerate JPEG chroma noise; the decision threshold
# (color_variant_min_delta) lives in config.toml
H_BINS = 12
HIST_SAMPLE_SIZE = 64


def hamming_hex(a: str, b: str) -> int:
    """Hamming distance between two same-length hex hash strings."""
    return (int(a, 16) ^ int(b, 16)).bit_count()


def compute_hashes(img: Image.Image) -> tuple[str, str]:
    """(phash, dhash) as 16-char hex strings."""
    return (
        str(imagehash.phash(img, hash_size=8)),
        str(imagehash.dhash(img, hash_size=8)),
    )


def hs_histogram(path: Path) -> np.ndarray:
    """Normalized saturation-weighted hue histogram of an image file.

    Weighting hue mass by saturation makes the metric invariant to global
    saturation/contrast edits (weights rescale uniformly, then normalize
    away) while a recolor moves hue mass wholesale — empirically a contrast
    boost scores ~0.01 and an R/B channel swap ~0.62. Known limitation of
    any global histogram: a colorway confined to a small fraction of the
    frame moves at most that fraction of mass and cannot trip the flag;
    those pairs still reach Phase 6 via the cosine adjudication band.
    """
    with Image.open(path) as img:
        hsv = img.convert("HSV").resize((HIST_SAMPLE_SIZE, HIST_SAMPLE_SIZE))
        arr = np.asarray(hsv).astype(float)
    hue_bin = (arr[:, :, 0].astype(int) * H_BINS // 256).ravel()
    saturation = arr[:, :, 1].ravel()
    hist = np.bincount(hue_bin, weights=saturation, minlength=H_BINS)
    return hist / (hist.sum() or 1.0)


def histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L1/2 distance between normalized histograms; 0 (same) to 1 (disjoint)."""
    return float(0.5 * np.abs(a - b).sum())


def _hash_missing(
    conn: sqlite3.Connection,
    fetcher: ImageFetcher,
    config: Config,
    scan_id: int,
) -> int:
    """Hash every active photo lacking hashes. Resumable: photos hashed on a
    previous run are skipped by the WHERE clause. Returns the number due.

    Fetch + hash run across ``perf.hash_workers`` threads (both the network
    fetch and the pHash/dHash release the GIL), while a single DbWriter thread
    owns every write to ``conn``. A failed photo is simply left un-hashed and
    retried next run — identical resumability to the old serial path.
    """
    todo = conn.execute(
        """
        SELECT id, drive_id FROM photos
        WHERE status = 'active' AND (phash IS NULL OR dhash IS NULL)
        ORDER BY id
        """
    ).fetchall()
    conn.execute("UPDATE scans SET total = ? WHERE id = ?", (len(todo), scan_id))
    conn.commit()

    def fetch_and_hash(row: sqlite3.Row) -> tuple[str, str]:
        path = fetcher.get(row["drive_id"], config.thumb_size)
        with Image.open(path) as img:
            return compute_hashes(img)

    perf = config.perf
    processed = 0
    with DbWriter(conn, perf.db_batch, perf.queue_maxsize) as writer:
        for row, result in parallel_map(fetch_and_hash, todo, perf.hash_workers):
            processed += 1
            if isinstance(result, Exception):
                traceback.print_exception(result)  # leave un-hashed; retried next run
            else:
                phash, dhash = result
                writer.put(
                    "UPDATE photos SET phash = ?, dhash = ? WHERE id = ?",
                    (phash, dhash, row["id"]),
                )
            writer.put(
                "UPDATE scans SET processed = ? WHERE id = ?", (processed, scan_id)
            )
    return len(todo)


def _candidate_pairs(
    rows: list[sqlite3.Row], radius: int
) -> list[tuple[sqlite3.Row, sqlite3.Row]]:
    """BK-tree over pHash: all photo pairs within `radius` Hamming distance."""
    by_hash: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        by_hash.setdefault(int(row["phash"], 16), []).append(row)
    tree = pybktree.BKTree(pybktree.hamming_distance, list(by_hash))
    pairs: list[tuple[sqlite3.Row, sqlite3.Row]] = []
    for phash_int, bucket in by_hash.items():
        pairs.extend(combinations(bucket, 2))
        for _dist, other in tree.find(phash_int, radius):
            if other <= phash_int:  # each cross-bucket pair once
                continue
            pairs.extend((a, b) for a in bucket for b in by_hash[other])
    return pairs


def _confirmed(a: sqlite3.Row, b: sqlite3.Row, config: Config) -> bool:
    """dHash confirmation to suppress pHash false positives; byte-identical
    pairs (same md5) belong to the exact tier, not here."""
    if a["md5"] is not None and a["md5"] == b["md5"]:
        return False
    return hamming_hex(a["dhash"], b["dhash"]) <= config.dhash_confirm_max


def _is_color_variant(
    fetcher: ImageFetcher, config: Config, drive_ids: list[str]
) -> bool:
    """Do the group's members share structure but differ in color?"""
    hists = [
        hs_histogram(fetcher.get(drive_id, config.thumb_size)) for drive_id in drive_ids
    ]
    delta = max(
        (histogram_distance(a, b) for a, b in combinations(hists, 2)),
        default=0.0,
    )
    return delta > config.color_variant_min_delta


def run_near(conn: sqlite3.Connection, fetcher: ImageFetcher, config: Config) -> int:
    """Hash, pair, confirm, group. Returns the scans row id."""
    resuming = stage_was_interrupted(conn, "near")
    scan_id = start_scan(conn, "near")
    try:
        if resuming:
            # re-hash the last few from a fresh fetch, in case the crash left a
            # partial write or truncated thumbnail at that boundary
            reprocess_tail(conn, "near", config.resume_overlap, config.cache_dir)
        n_hashed = _hash_missing(conn, fetcher, config, scan_id)

        rows = conn.execute(
            """
            SELECT id, drive_id, md5, phash, dhash FROM photos
            WHERE status = 'active' AND phash IS NOT NULL AND dhash IS NOT NULL
            """
        ).fetchall()
        by_id = {row["id"]: row for row in rows}

        uf = UnionFind()
        for a, b in _candidate_pairs(rows, config.near_hamming_max):
            if _confirmed(a, b, config):
                uf.union(a["id"], b["id"])

        # compute colorway flags BEFORE opening the rebuild transaction:
        # fetcher.get() records thumb_path on a second DB connection, which
        # would deadlock against this connection's open write transaction
        clusters = uf.clusters()
        variant_flags = [
            _is_color_variant(
                fetcher,
                config,
                [by_id[photo_id]["drive_id"] for photo_id in cluster],
            )
            for cluster in clusters
        ]

        rebuild_groups(conn, "near")
        for cluster, is_variant in zip(clusters, variant_flags, strict=True):
            anchor = by_id[cluster[0]]
            cur = conn.execute(
                "INSERT INTO groups (tier, color_variant, created_at) "
                "VALUES ('near', ?, ?)",
                (1 if is_variant else 0, now()),
            )
            group_id = int(cur.lastrowid)
            for photo_id in cluster:
                conn.execute(
                    "INSERT INTO group_members (group_id, photo_id, score) "
                    "VALUES (?, ?, ?)",
                    (
                        group_id,
                        photo_id,
                        hamming_hex(by_id[photo_id]["phash"], anchor["phash"]),
                    ),
                )
        conn.commit()
        finish_scan(conn, scan_id, total=n_hashed)
    except BaseException as exc:
        conn.rollback()
        fail_scan(conn, scan_id, f"{type(exc).__name__}: {exc}")
        raise
    return scan_id
