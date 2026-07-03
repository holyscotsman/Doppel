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
from doppel.jobs import fail_scan, finish_scan, now, start_scan
from doppel.stages.exact import rebuild_groups
from doppel.stages.grouping import UnionFind

# structural constants of the colorway metric (not tunable thresholds):
# coarse bins tolerate JPEG chroma noise; the decision threshold
# (color_variant_min_delta) lives in config.toml
H_BINS = 12
S_BINS = 4
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
    """Normalized 2D hue/saturation histogram of an image file."""
    with Image.open(path) as img:
        hsv = img.convert("HSV").resize((HIST_SAMPLE_SIZE, HIST_SAMPLE_SIZE))
        arr = np.asarray(hsv).astype(int)
    idx = (arr[:, :, 0] * H_BINS // 256) * S_BINS + (arr[:, :, 1] * S_BINS // 256)
    hist = np.bincount(idx.ravel(), minlength=H_BINS * S_BINS).astype(float)
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
    previous run are skipped by the WHERE clause. Returns the number due."""
    todo = conn.execute(
        """
        SELECT id, drive_id FROM photos
        WHERE status = 'active' AND (phash IS NULL OR dhash IS NULL)
        ORDER BY id
        """
    ).fetchall()
    conn.execute("UPDATE scans SET total = ? WHERE id = ?", (len(todo), scan_id))
    conn.commit()
    for i, row in enumerate(todo, start=1):
        try:
            path = fetcher.get(row["drive_id"], config.thumb_size)
            with Image.open(path) as img:
                phash, dhash = compute_hashes(img)
        except Exception:
            # leave un-hashed: the next run retries this photo
            traceback.print_exc()
        else:
            conn.execute(
                "UPDATE photos SET phash = ?, dhash = ? WHERE id = ?",
                (phash, dhash, row["id"]),
            )
        conn.execute("UPDATE scans SET processed = ? WHERE id = ?", (i, scan_id))
        conn.commit()
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


def _flag_color_variants(
    conn: sqlite3.Connection,
    fetcher: ImageFetcher,
    config: Config,
    group_id: int,
    drive_ids: list[str],
) -> None:
    """Badge groups whose members share structure but differ in color."""
    hists = [
        hs_histogram(fetcher.get(drive_id, config.thumb_size)) for drive_id in drive_ids
    ]
    delta = max(
        (histogram_distance(a, b) for a, b in combinations(hists, 2)),
        default=0.0,
    )
    if delta > config.color_variant_min_delta:
        conn.execute("UPDATE groups SET color_variant = 1 WHERE id = ?", (group_id,))


def run_near(conn: sqlite3.Connection, fetcher: ImageFetcher, config: Config) -> int:
    """Hash, pair, confirm, group. Returns the scans row id."""
    scan_id = start_scan(conn, "near")
    try:
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

        rebuild_groups(conn, "near")
        for cluster in uf.clusters():
            anchor = by_id[cluster[0]]
            cur = conn.execute(
                "INSERT INTO groups (tier, created_at) VALUES ('near', ?)",
                (now(),),
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
            _flag_color_variants(
                conn,
                fetcher,
                config,
                group_id,
                [by_id[photo_id]["drive_id"] for photo_id in cluster],
            )
        conn.commit()
        finish_scan(conn, scan_id, total=n_hashed)
    except BaseException as exc:
        conn.rollback()
        fail_scan(conn, scan_id, f"{type(exc).__name__}: {exc}")
        raise
    return scan_id
