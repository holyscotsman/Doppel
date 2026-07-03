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
