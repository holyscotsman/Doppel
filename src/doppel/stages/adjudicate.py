"""Phase 6 — VLM adjudication of borderline duplicates.

Stages 1-3 nominate candidates with cheap math; the VLM only rules on:
- similar pairs whose cosine falls in [adjudicate_band_min,
  similar_cosine_min) — too weak to auto-group, too strong to ignore
- pairs inside near-tier groups flagged color_variant

Verdicts become tier-'vlm' groups ('variant' sets color_variant = 1); the
human still decides in the UI.
"""

from __future__ import annotations

import json
import sqlite3
from itertools import combinations
from pathlib import Path

from doppel.config import Config
from doppel.db import ensure_vec_schema
from doppel.drive import ImageFetcher
from doppel.jobs import fail_scan, finish_scan, now, start_scan
from doppel.stages.exact import rebuild_groups
from doppel.stages.grouping import UnionFind
from doppel.stages.parallel import DbWriter, parallel_map
from doppel.stages.similar import K_NEIGHBORS, load_vectors
from doppel.vlm import PROMPTS_DIR, VlmClient, latest_prompt

ADJUDICATE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["same", "near", "variant", "different"],
        },
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}

GROUPING_VERDICTS = ("same", "near", "variant")


def _band_pairs(conn: sqlite3.Connection, config: Config) -> set[tuple[int, int]]:
    """Similar pairs in the adjudication band, from stored vectors.

    Band pairs are farther than auto-group pairs, so they are the first
    crowded out of a photo's top-k by any burst of closer shots — keep
    pairs from whichever direction's kNN finds them (normalized ordering),
    never assume symmetry.
    """
    ensure_vec_schema(conn)
    vectors = load_vectors(conn)
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
                continue
            if config.ollama.adjudicate_band_min <= cosine < config.similar_cosine_min:
                pairs.add((min(photo_id, other), max(photo_id, other)))
    return pairs


def _color_variant_pairs(conn: sqlite3.Connection) -> set[tuple[int, int]]:
    """All member pairs of near-tier groups flagged color_variant."""
    groups: dict[int, list[int]] = {}
    for row in conn.execute(
        """
        SELECT g.id, m.photo_id FROM groups g
        JOIN group_members m ON m.group_id = g.id
        JOIN photos p ON p.id = m.photo_id
        WHERE g.tier = 'near' AND g.color_variant = 1 AND p.status = 'active'
        """
    ):
        groups.setdefault(row["id"], []).append(row["photo_id"])
    pairs: set[tuple[int, int]] = set()
    for members in groups.values():
        pairs.update(combinations(sorted(members), 2))
    return pairs


def _rebuild_vlm_groups(
    conn: sqlite3.Connection, model: str, prompt_version: str
) -> None:
    """Rebuild tier-'vlm' groups from stored verdicts of the current model +
    prompt version. same/near/variant verdicts group; any variant edge in a
    cluster flags the group color_variant."""
    rows = conn.execute(
        """
        SELECT v.photo_id, v.photo_id_b, v.verdict FROM vlm_results v
        JOIN photos a ON a.id = v.photo_id
        JOIN photos b ON b.id = v.photo_id_b
        WHERE v.task = 'adjudicate' AND v.model = ? AND v.prompt_version = ?
          AND a.status = 'active' AND b.status = 'active'
        """,
        (model, prompt_version),
    ).fetchall()
    uf = UnionFind()
    variant_edges: list[tuple[int, int]] = []
    for row in rows:
        if row["verdict"] in GROUPING_VERDICTS:
            uf.union(row["photo_id"], row["photo_id_b"])
            if row["verdict"] == "variant":
                variant_edges.append((row["photo_id"], row["photo_id_b"]))

    rebuild_groups(conn, "vlm")
    for cluster in uf.clusters():
        members = set(cluster)
        color_variant = any(a in members and b in members for a, b in variant_edges)
        cur = conn.execute(
            "INSERT INTO groups (tier, color_variant, created_at) VALUES ('vlm', ?, ?)",
            (1 if color_variant else 0, now()),
        )
        group_id = int(cur.lastrowid)
        for photo_id in cluster:
            conn.execute(
                "INSERT INTO group_members (group_id, photo_id) VALUES (?, ?)",
                (group_id, photo_id),
            )


def run_adjudicate(
    conn: sqlite3.Connection,
    fetcher: ImageFetcher,
    vlm: VlmClient,
    config: Config,
    prompts_dir: Path | str = PROMPTS_DIR,
) -> int:
    """Adjudicate borderline pairs. Resumable: pairs already ruled on by the
    current model + prompt version are skipped; bumping the prompt version
    re-adjudicates without clobbering prior results."""
    # No resume-overlap here: a resumed adjudicate simply re-asks pairs that
    # still lack a verdict (the `done` filter below). A verdict is validated
    # JSON, not a downloaded byte stream, so there's nothing to "re-fetch", and
    # deleting one whose pair is no longer a candidate would silently lose it.
    scan_id = start_scan(conn, "adjudicate")
    try:
        prompt, version = latest_prompt("adjudicate", prompts_dir)
        model = config.ollama.model
        candidates = sorted(_band_pairs(conn, config) | _color_variant_pairs(conn))
        done = {
            (row["photo_id"], row["photo_id_b"])
            for row in conn.execute(
                "SELECT photo_id, photo_id_b FROM vlm_results "
                "WHERE task = 'adjudicate' AND model = ? AND prompt_version = ?",
                (model, version),
            )
        }
        todo = [pair for pair in candidates if pair not in done]
        conn.execute("UPDATE scans SET total = ? WHERE id = ?", (len(todo), scan_id))
        conn.commit()

        drive_ids = {
            row["id"]: row["drive_id"]
            for row in conn.execute("SELECT id, drive_id FROM photos")
        }

        def adjudicate_pair(pair: tuple[int, int]) -> dict:
            a, b = pair
            image_a = fetcher.get(drive_ids[a], config.thumb_size).read_bytes()
            image_b = fetcher.get(drive_ids[b], config.thumb_size).read_bytes()
            return vlm.chat_json(prompt, [image_a, image_b], ADJUDICATE_SCHEMA)

        # Fetch + VLM run across perf.adjudicate_workers threads (kept low —
        # Ollama is one local server); a single DbWriter owns every insert. A
        # pair whose fetch/VLM raises is left un-ruled and retried next run.
        perf = config.perf
        processed = 0
        with DbWriter(conn, perf.db_batch, perf.queue_maxsize) as writer:
            for (a, b), result in parallel_map(
                adjudicate_pair, todo, perf.adjudicate_workers
            ):
                processed += 1
                if isinstance(result, Exception):
                    raise result  # fail the scan: an interrupt/VLM error is real
                verdict = result.get("verdict")
                if verdict not in (*GROUPING_VERDICTS, "different"):
                    verdict = None  # schema should prevent this; stay honest
                writer.put(
                    """
                    INSERT INTO vlm_results
                      (task, photo_id, photo_id_b, model, prompt_version,
                       response, verdict, created_at)
                    VALUES ('adjudicate', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (a, b, model, version, json.dumps(result), verdict, now()),
                )
                writer.put(
                    "UPDATE scans SET processed = ? WHERE id = ?", (processed, scan_id)
                )

        _rebuild_vlm_groups(conn, model, version)
        conn.commit()
        finish_scan(conn, scan_id, total=len(todo))
    except BaseException as exc:
        conn.rollback()
        fail_scan(conn, scan_id, f"{type(exc).__name__}: {exc}")
        raise
    return scan_id
