"""Fake Drive client for tests. Tests must never hit the real Drive API."""

from __future__ import annotations

from typing import Any


def make_file(
    drive_id: str,
    name: str = "photo.jpg",
    mime_type: str = "image/jpeg",
    size: int | None = 1000,
    md5: str | None = None,
    width: int | None = 800,
    height: int | None = 600,
    thumbnail_link: str | None = None,
) -> dict[str, Any]:
    """Build a Drive files.list item as the API returns it (size is a string)."""
    file: dict[str, Any] = {
        "id": drive_id,
        "name": name,
        "mimeType": mime_type,
        "createdTime": "2024-01-01T00:00:00Z",
        "modifiedTime": "2024-01-02T00:00:00Z",
    }
    if size is not None:
        file["size"] = str(size)
    if md5 is not None:
        file["md5Checksum"] = md5
    if width is not None or height is not None:
        file["imageMediaMetadata"] = {"width": width, "height": height}
    if thumbnail_link is not None:
        file["thumbnailLink"] = thumbnail_link
    return file


class FakeDriveClient:
    """Serves canned files in pages, mimicking files.list pagination."""

    def __init__(self, files: list[dict[str, Any]], page_size: int = 2) -> None:
        self.files = files
        self.page_size = page_size
        self.pages_served: list[str | None] = []
        self.fail_after_pages: int | None = None

    def list_images_page(self, page_token: str | None = None) -> dict[str, Any]:
        failing = (
            self.fail_after_pages is not None
            and len(self.pages_served) >= self.fail_after_pages
        )
        if failing:
            raise RuntimeError("simulated Drive API failure")
        self.pages_served.append(page_token)
        start = int(page_token) if page_token else 0
        end = start + self.page_size
        page: dict[str, Any] = {"files": self.files[start:end]}
        if end < len(self.files):
            page["nextPageToken"] = str(end)
        return page


class FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"") -> None:
        self.status_code = status_code
        self.content = content


class FakeSession:
    """Returns queued responses in order and records requested URLs."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[str] = []

    def get(self, url: str) -> FakeResponse:
        self.requests.append(url)
        return self.responses.pop(0)


class FakeThumbClient:
    """Fake of the Drive metadata/media calls the fetcher uses."""

    def __init__(self, link: str | None = None, file_bytes: bytes = b"") -> None:
        self.link = link
        self.file_bytes = file_bytes
        self.link_requests: list[str] = []
        self.download_requests: list[str] = []

    def get_thumbnail_link(self, drive_id: str) -> str | None:
        self.link_requests.append(drive_id)
        return self.link

    def download_file(self, drive_id: str) -> bytes:
        self.download_requests.append(drive_id)
        return self.file_bytes


def jpeg_bytes(size: tuple[int, int] = (32, 32), color: str = "red") -> bytes:
    """A real JPEG for tests, generated with Pillow."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


class FakeImageFetcher:
    """Serves generated JPEGs from the cache dir without any network.

    `images` maps drive_id -> jpeg bytes (default: a flat red square);
    map a drive_id to an Exception instance to simulate a fetch failure.
    """

    def __init__(self, cache_dir, images: dict[str, object] | None = None) -> None:
        from pathlib import Path

        self.cache_dir = Path(cache_dir)
        self.images = images or {}
        self.calls: list[tuple[str, int | str]] = []

    def get(self, drive_id: str, size: int | str = 512):
        self.calls.append((drive_id, size))
        source = self.images.get(drive_id)
        if isinstance(source, BaseException):
            raise source
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{drive_id}_{size}.jpg"
        if not path.exists():
            path.write_bytes(source if source is not None else jpeg_bytes())
        return path


def insert_photo(
    conn,
    drive_id: str,
    name: str = "photo.jpg",
    md5: str | None = None,
    size: int | None = 1000,
    status: str = "active",
    thumbnail_link: str | None = None,
    width: int | None = 800,
    height: int | None = 600,
) -> int:
    """Insert a photos row directly (bypassing sync) and return its id."""
    cur = conn.execute(
        """
        INSERT INTO photos (drive_id, name, mime_type, size, md5, width, height,
                            thumbnail_link, status)
        VALUES (?, ?, 'image/jpeg', ?, ?, ?, ?, ?, ?)
        """,
        (drive_id, name, size, md5, width, height, thumbnail_link, status),
    )
    conn.commit()
    return int(cur.lastrowid)


def drive_id_from_path(path) -> str:
    """Recover the drive_id from a cache path like cache/{drive_id}_{size}.jpg."""
    from pathlib import Path

    return Path(path).name.rsplit("_", 1)[0]


class FakeEmbedder:
    """Returns canned vectors keyed by drive_id; records embed calls."""

    def __init__(self, vectors: dict[str, object]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed(self, paths):
        import numpy as np

        ids = [drive_id_from_path(p) for p in paths]
        self.calls.append(ids)
        return np.stack([self.vectors[i] for i in ids]).astype(np.float32)
