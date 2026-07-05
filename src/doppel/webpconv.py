"""Post-scan WebP -> lossless PNG conversion.

For every WebP the scan recorded, write a lossless PNG beside it in the same
Drive folder and move the original WebP into a trash folder. PNG is lossless, so
the WebP's decoded pixels (including any alpha channel) are preserved exactly —
no second generation of quality loss. Animated WebPs are skipped (a single PNG
frame would drop the animation).

This is the app's only UNATTENDED Drive write, so it is opt-in
(`[webp] convert_after_scan`) and needs the owner sign-in — a service account is
not the owner of the user's files and cannot create or move them. It never
deletes: originals only move (reversible), and every result is recorded in
`webp_conversions` so re-runs skip already-processed files.
"""

from __future__ import annotations

import io
import logging
import sqlite3
from datetime import UTC, datetime
from typing import Protocol

log = logging.getLogger("doppel.webp")


def convert_webp_to_png(data: bytes) -> tuple[bytes | None, str]:
    """Losslessly re-encode WebP image bytes as PNG, preserving any alpha.

    Returns (png_bytes, "ok"), or (None, reason) when the image can't or
    shouldn't be converted: an animated WebP (PNG holds a single frame), or
    bytes Pillow can't decode. The PNG stores the decoded pixels exactly.
    """
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # noqa: BLE001 — any decode failure is a skip
        return None, f"unreadable image ({exc.__class__.__name__})"
    if getattr(img, "is_animated", False) or getattr(img, "n_frames", 1) > 1:
        return None, "animated WebP (PNG can't hold animation)"
    # keep alpha when present; otherwise a plain RGB PNG
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
    else:
        img = img.convert("RGB")
    out = io.BytesIO()
    try:
        img.save(out, format="PNG", optimize=True)
    except Exception as exc:  # noqa: BLE001
        return None, f"PNG encode failed ({exc.__class__.__name__})"
    return out.getvalue(), "ok"


def png_name_for(webp_name: str) -> str:
    """The PNG filename for a WebP: swap a trailing .webp, else just append."""
    base = webp_name
    if base.lower().endswith(".webp"):
        base = base[:-5]
    return base + ".png"


class WriteDriveClient(Protocol):
    """The owner-scoped Drive operations the converter needs. Tests use a fake."""

    def download_file(self, drive_id: str) -> bytes: ...
    def root_folder_id(self) -> str: ...
    def ensure_child_folder(self, name: str, parent_id: str | None) -> str: ...
    def find_child(self, name: str, parent_id: str | None) -> str | None: ...
    def upload_file(
        self, name: str, parent_id: str | None, data: bytes, mime_type: str
    ) -> str: ...
    def move_file(self, drive_id: str, add_parent: str, remove_parent: str) -> None: ...


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _record(
    conn: sqlite3.Connection,
    source: str,
    png: str | None,
    status: str,
    reason: str | None,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO webp_conversions "
        "(source_drive_id, png_drive_id, status, reason, converted_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (source, png, status, reason, _now()),
    )
    conn.commit()


def pending_webps(conn: sqlite3.Connection, trash_folder: str) -> list[sqlite3.Row]:
    """Active WebP files not yet processed and not already sitting in the trash
    folder (largest first, so the biggest space wins convert first)."""
    # Escape LIKE metacharacters so the folder guard matches the trash folder
    # LITERALLY. A bare "_" is a LIKE wildcard, so an unescaped "WEBP_Trash"
    # pattern would also exclude an unrelated folder like "WEBP Trash".
    esc = trash_folder.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return conn.execute(
        """
        SELECT drive_id, name, parent_id, folder_path, size
        FROM photos
        WHERE mime_type = 'image/webp' AND status = 'active'
          AND (folder_path IS NULL OR folder_path NOT LIKE ? ESCAPE '\\')
          AND drive_id NOT IN (SELECT source_drive_id FROM webp_conversions)
        ORDER BY size DESC, name
        """,
        (f"%{esc}%",),
    ).fetchall()


def convert_library_webp(
    conn: sqlite3.Connection,
    client: WriteDriveClient,
    *,
    trash_folder: str = "WEBP_Trash",
) -> dict[str, int]:
    """Convert every not-yet-processed WebP to a lossless PNG in the same folder
    and move the original into `trash_folder`. Idempotent (skips recorded files)
    and reversible (originals move, never delete). `client` must be write-scoped
    (the owner). Returns a summary of counts.
    """
    rows = pending_webps(conn, trash_folder)
    summary = {"total": len(rows), "converted": 0, "skipped": 0, "failed": 0}
    if not rows:
        return summary

    trash_id: str | None = None  # created lazily on the first real conversion
    root_id: str | None = None  # resolved lazily for root-level files

    for r in rows:
        webp_id, name = r["drive_id"], (r["name"] or r["drive_id"])
        try:
            data = client.download_file(webp_id)
        except Exception as exc:  # noqa: BLE001 — transient: retry next scan
            log.warning("webp: download failed for %s: %s", name, exc)
            summary["failed"] += 1
            continue

        png, reason = convert_webp_to_png(data)
        if png is None:
            _record(conn, webp_id, None, "skipped", reason)
            summary["skipped"] += 1
            log.info("webp: skipped %s (%s)", name, reason)
            continue

        try:
            parent = r["parent_id"]
            if parent is None:
                root_id = root_id or client.root_folder_id()
                parent = root_id
            target = png_name_for(name)
            if client.find_child(target, parent) is not None:
                _record(conn, webp_id, None, "skipped", "a PNG of that name exists")
                summary["skipped"] += 1
                log.info("webp: skipped %s (target %s already exists)", name, target)
                continue
            if trash_id is None:
                trash_id = client.ensure_child_folder(trash_folder, None)
            png_id = client.upload_file(target, parent, png, "image/png")
            client.move_file(webp_id, trash_id, parent)
        except Exception as exc:  # noqa: BLE001 — transient: retry next scan
            log.warning("webp: write failed for %s: %s", name, exc)
            summary["failed"] += 1
            continue

        _record(conn, webp_id, png_id, "converted", None)
        summary["converted"] += 1
        log.info("webp: converted %s -> %s", name, target)

    log.info(
        "webp: pass complete — %d converted, %d skipped, %d failed (of %d)",
        summary["converted"],
        summary["skipped"],
        summary["failed"],
        summary["total"],
    )
    return summary
