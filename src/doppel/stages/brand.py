"""Phase 7 — brand tagging of deduped keepers.

Runs after review, over active photos without a 'trash' decision. Fetches
ORIGINAL resolution via ImageFetcher's orig path — logos need pixels;
thumbnails are insufficient here. This is the only consumer of original
bytes in the app.

Human corrections (tags.source = 'human') are never overwritten and their
photos are never re-sent to the VLM.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from doppel.config import Config
from doppel.drive import ImageFetcher
from doppel.jobs import fail_scan, finish_scan, now, start_scan
from doppel.vlm import PROMPTS_DIR, VlmClient, latest_prompt

BRAND_SCHEMA = {
    "type": "object",
    "properties": {
        "brand": {"type": "string"},
        "evidence": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["brand", "evidence", "confidence"],
}


def run_brand(
    conn: sqlite3.Connection,
    fetcher: ImageFetcher,
    vlm: VlmClient,
    config: Config,
    prompts_dir: Path | str = PROMPTS_DIR,
) -> int:
    """Tag every non-trashed active photo with a brand. Resumable: photos
    already tagged by the current model + prompt version are skipped, as are
    photos with a human-corrected tag."""
    scan_id = start_scan(conn, "brand")
    try:
        prompt, version = latest_prompt("brand", prompts_dir)
        model = config.ollama.model
        todo = conn.execute(
            """
            SELECT id, drive_id FROM photos
            WHERE status = 'active'
              AND id NOT IN
                (SELECT photo_id FROM decisions WHERE action = 'trash')
              AND id NOT IN
                (SELECT photo_id FROM tags
                 WHERE kind = 'brand' AND source = 'human')
              AND id NOT IN
                (SELECT photo_id FROM vlm_results
                 WHERE task = 'brand' AND model = ? AND prompt_version = ?)
            ORDER BY id
            """,
            (model, version),
        ).fetchall()
        conn.execute("UPDATE scans SET total = ? WHERE id = ?", (len(todo), scan_id))
        conn.commit()

        for i, row in enumerate(todo, start=1):
            image = fetcher.get(row["drive_id"], "orig").read_bytes()
            result = vlm.chat_json(prompt, [image], BRAND_SCHEMA)
            brand = str(result.get("brand", "unknown")) or "unknown"
            confidence = result.get("confidence")
            conn.execute(
                """
                INSERT INTO vlm_results
                  (task, photo_id, model, prompt_version, response,
                   verdict, confidence, created_at)
                VALUES ('brand', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    model,
                    version,
                    json.dumps(result),
                    brand,
                    confidence,
                    now(),
                ),
            )
            conn.execute(
                """
                INSERT INTO tags
                  (photo_id, kind, value, confidence, source, created_at)
                VALUES (?, 'brand', ?, ?, 'vlm', ?)
                ON CONFLICT(photo_id, kind) DO UPDATE SET
                  value = excluded.value,
                  confidence = excluded.confidence,
                  source = 'vlm',
                  created_at = excluded.created_at
                WHERE tags.source != 'human'
                """,
                (row["id"], brand, confidence, now()),
            )
            conn.execute("UPDATE scans SET processed = ? WHERE id = ?", (i, scan_id))
            conn.commit()  # per-photo commit: interrupt-safe resume
        finish_scan(conn, scan_id, total=len(todo))
    except BaseException as exc:
        conn.rollback()
        fail_scan(conn, scan_id, f"{type(exc).__name__}: {exc}")
        raise
    return scan_id


def correct_brand(conn: sqlite3.Connection, photo_id: int, value: str) -> None:
    """Record a human correction; re-runs never overwrite it."""
    conn.execute(
        """
        INSERT INTO tags (photo_id, kind, value, confidence, source, created_at)
        VALUES (?, 'brand', ?, 1.0, 'human', ?)
        ON CONFLICT(photo_id, kind) DO UPDATE SET
          value = excluded.value,
          confidence = 1.0,
          source = 'human',
          created_at = excluded.created_at
        """,
        (photo_id, value, now()),
    )
    conn.commit()
