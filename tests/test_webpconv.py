"""Post-scan WebP -> lossless PNG conversion: the pure converter, and the
idempotent/reversible orchestration (originals move to a trash folder, never
deleted; re-runs skip processed files)."""

from __future__ import annotations

import io

from PIL import Image

from doppel.db import connect
from doppel.webpconv import (
    convert_library_webp,
    convert_webp_to_png,
    png_name_for,
)
from tests.fakes import insert_photo


def _webp(mode: str = "RGB", color=(200, 120, 60), size=(8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, "WEBP", lossless=True)
    return buf.getvalue()


def _animated_webp() -> bytes:
    frames = [Image.new("RGB", (8, 8), (i * 40, 0, 0)) for i in range(3)]
    buf = io.BytesIO()
    frames[0].save(buf, "WEBP", save_all=True, append_images=frames[1:], duration=80)
    return buf.getvalue()


def _webp_photo(conn, drive_id, name, parent_id=None, folder_path=None) -> int:
    pid = insert_photo(
        conn, drive_id, name=name, parent_id=parent_id, folder_path=folder_path
    )
    conn.execute("UPDATE photos SET mime_type = 'image/webp' WHERE id = ?", (pid,))
    conn.commit()
    return pid


class FakeWriteClient:
    """Records the converter's Drive writes; never deletes anything."""

    def __init__(self, originals: dict[str, bytes], existing=None) -> None:
        self.originals = originals
        self.existing = set(existing or [])  # {(name, parent)} that already exist
        self.uploaded: list[dict] = []
        self.moved: list[tuple] = []
        self.folders: dict = {}
        self._n = 0

    def download_file(self, drive_id: str) -> bytes:
        return self.originals[drive_id]

    def root_folder_id(self) -> str:
        return "ROOT"

    def ensure_child_folder(self, name, parent_id) -> str:
        return self.folders.setdefault((name, parent_id), f"folder:{name}")

    def find_child(self, name, parent_id):
        return "exists" if (name, parent_id) in self.existing else None

    def upload_file(self, name, parent_id, data, mime_type) -> str:
        self._n += 1
        pid = f"png-{self._n}"
        self.uploaded.append(
            {
                "name": name,
                "parent": parent_id,
                "mime": mime_type,
                "id": pid,
                "data": data,
            }
        )
        return pid

    def move_file(self, drive_id, add_parent, remove_parent) -> None:
        self.moved.append((drive_id, add_parent, remove_parent))


# ---- pure converter -----------------------------------------------------


def test_rgb_webp_becomes_png_with_pixels_intact() -> None:
    png, reason = convert_webp_to_png(_webp("RGB", (10, 20, 30)))
    assert reason == "ok"
    out = Image.open(io.BytesIO(png))
    assert out.format == "PNG"
    assert out.convert("RGB").getpixel((0, 0)) == (10, 20, 30)  # lossless


def test_transparency_is_preserved() -> None:
    png, reason = convert_webp_to_png(_webp("RGBA", (10, 20, 30, 128)))
    assert reason == "ok"
    out = Image.open(io.BytesIO(png))
    assert out.mode == "RGBA"
    assert out.getpixel((0, 0)) == (10, 20, 30, 128)  # alpha kept exactly


def test_animated_webp_is_skipped() -> None:
    png, reason = convert_webp_to_png(_animated_webp())
    assert png is None
    assert "animated" in reason.lower()


def test_unreadable_bytes_are_skipped() -> None:
    png, reason = convert_webp_to_png(b"not an image")
    assert png is None
    assert "unreadable" in reason.lower()


def test_png_name_swaps_the_extension() -> None:
    assert png_name_for("photo.webp") == "photo.png"
    assert png_name_for("PHOTO.WEBP") == "PHOTO.png"
    assert png_name_for("noext") == "noext.png"


# ---- orchestration ------------------------------------------------------


def test_converts_uploads_png_and_moves_original(config) -> None:
    conn = connect(config.db_path)
    _webp_photo(conn, "w1", "a.webp", parent_id="folderA")
    client = FakeWriteClient({"w1": _webp("RGB")})

    summary = convert_library_webp(conn, client, trash_folder="WEBP_Trash")

    assert summary == {"total": 1, "converted": 1, "skipped": 0, "failed": 0}
    # PNG written into the SAME folder as the webp
    assert client.uploaded[0]["name"] == "a.png"
    assert client.uploaded[0]["parent"] == "folderA"
    assert client.uploaded[0]["mime"] == "image/png"
    # original moved into the trash folder (added there, removed from folderA)
    assert client.moved == [("w1", "folder:WEBP_Trash", "folderA")]
    row = conn.execute(
        "SELECT status, png_drive_id FROM webp_conversions WHERE source_drive_id='w1'"
    ).fetchone()
    assert row["status"] == "converted" and row["png_drive_id"] == "png-1"
    conn.close()


def test_second_run_is_idempotent(config) -> None:
    conn = connect(config.db_path)
    _webp_photo(conn, "w1", "a.webp", parent_id="folderA")
    client = FakeWriteClient({"w1": _webp("RGB")})
    convert_library_webp(conn, client, trash_folder="WEBP_Trash")

    client2 = FakeWriteClient({"w1": _webp("RGB")})
    summary = convert_library_webp(conn, client2, trash_folder="WEBP_Trash")
    assert summary["total"] == 0  # already recorded -> nothing to do
    assert client2.uploaded == [] and client2.moved == []
    conn.close()


def test_animated_is_recorded_skipped_and_not_retried(config) -> None:
    conn = connect(config.db_path)
    _webp_photo(conn, "anim", "clip.webp", parent_id="f")
    client = FakeWriteClient({"anim": _animated_webp()})
    summary = convert_library_webp(conn, client, trash_folder="WEBP_Trash")
    assert summary["skipped"] == 1 and summary["converted"] == 0
    assert client.uploaded == [] and client.moved == []
    row = conn.execute(
        "SELECT status, reason FROM webp_conversions WHERE source_drive_id='anim'"
    ).fetchone()
    assert row["status"] == "skipped" and "animated" in row["reason"].lower()
    conn.close()


def test_download_failure_is_transient_not_recorded(config) -> None:
    conn = connect(config.db_path)
    _webp_photo(conn, "w1", "a.webp", parent_id="f")
    client = FakeWriteClient({})  # download_file raises KeyError

    summary = convert_library_webp(conn, client, trash_folder="WEBP_Trash")
    assert summary["failed"] == 1 and summary["converted"] == 0
    # NOT recorded -> a later scan will retry it
    assert (
        conn.execute("SELECT COUNT(*) AS n FROM webp_conversions").fetchone()["n"] == 0
    )
    conn.close()


def test_existing_target_png_is_not_duplicated(config) -> None:
    conn = connect(config.db_path)
    _webp_photo(conn, "w1", "a.webp", parent_id="folderA")
    client = FakeWriteClient({"w1": _webp("RGB")}, existing={("a.png", "folderA")})
    summary = convert_library_webp(conn, client, trash_folder="WEBP_Trash")
    assert summary["skipped"] == 1 and client.uploaded == [] and client.moved == []
    conn.close()


def test_originals_already_in_trash_folder_are_ignored(config) -> None:
    conn = connect(config.db_path)
    _webp_photo(conn, "w1", "a.webp", folder_path="My Drive / WEBP_Trash")
    client = FakeWriteClient({"w1": _webp("RGB")})
    summary = convert_library_webp(conn, client, trash_folder="WEBP_Trash")
    assert summary["total"] == 0  # skipped by the folder-path guard
    conn.close()


def test_lookalike_folder_is_not_wildcard_excluded(config) -> None:
    """The '_' in 'WEBP_Trash' must be treated literally, not as a LIKE wildcard,
    so a user folder named 'WEBP Trash' (space) is still converted."""
    conn = connect(config.db_path)
    _webp_photo(
        conn, "w1", "a.webp", parent_id="f", folder_path="My Drive / WEBP Trash"
    )
    client = FakeWriteClient({"w1": _webp("RGB")})
    summary = convert_library_webp(conn, client, trash_folder="WEBP_Trash")
    assert summary["converted"] == 1  # NOT wrongly excluded by the '_' wildcard
    conn.close()
