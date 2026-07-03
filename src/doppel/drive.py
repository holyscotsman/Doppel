"""Google Drive integration: OAuth, image listing.

Scope is drive.readonly — v1 never modifies anything in Drive.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

LIST_QUERY = "mimeType contains 'image/' and trashed = false"
LIST_FIELDS = (
    "nextPageToken, files(id, name, mimeType, size, md5Checksum, "
    "imageMediaMetadata(width, height), createdTime, modifiedTime, thumbnailLink)"
)
PAGE_SIZE = 1000


class DriveClient(Protocol):
    """The subset of the Drive API the sync job needs. Tests use a fake."""

    def list_images_page(self, page_token: str | None = None) -> dict[str, Any]:
        """One page of files.list: {'files': [...], 'nextPageToken': str | absent}."""
        ...


class CredentialsRequired(RuntimeError):
    """Interactive OAuth is needed but the caller cannot run it."""


def get_credentials(
    credentials_path: Path | str = "credentials.json",
    token_path: Path | str = "token.json",
    allow_interactive: bool = True,
) -> Any:
    """Run the OAuth installed-app flow, caching the refresh token.

    allow_interactive=False (web worker threads, request handlers) raises
    CredentialsRequired instead of opening a browser consent flow — an
    interactive flow on a background thread would block it forever if the
    user misses the tab. Never log or print the credential contents.
    """
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    credentials_path = Path(credentials_path)
    token_path = Path(token_path)

    creds = None
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except (ValueError, KeyError):
            creds = None  # corrupt token.json — fall through to re-auth
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            token_path.write_text(creds.to_json())
        except RefreshError:
            creds = None  # token revoked/expired — fall through to re-auth
    if not creds or not creds.valid:
        if not allow_interactive:
            raise CredentialsRequired(
                "Drive authorization required — run `make scan` once in a "
                "terminal to complete the OAuth consent flow."
            )
        if not credentials_path.exists():
            raise FileNotFoundError(
                f"{credentials_path} not found. Create an OAuth client ID "
                "(Desktop app) in Google Cloud Console, enable the Drive API, "
                "and download it to the repo root as credentials.json."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
        creds = flow.run_local_server(port=0, open_browser=True)
        token_path.write_text(creds.to_json())
    return creds


class GoogleDriveClient:
    """Real Drive client. Everything network-facing lives behind this."""

    def __init__(self, credentials: Any) -> None:
        from googleapiclient.discovery import build

        self._service = build("drive", "v3", credentials=credentials)

    def list_images_page(self, page_token: str | None = None) -> dict[str, Any]:
        return (
            self._service.files()
            .list(
                q=LIST_QUERY,
                pageSize=PAGE_SIZE,
                fields=LIST_FIELDS,
                pageToken=page_token,
            )
            .execute()
        )

    def get_thumbnail_link(self, drive_id: str) -> str | None:
        meta = (
            self._service.files().get(fileId=drive_id, fields="thumbnailLink").execute()
        )
        return meta.get("thumbnailLink")

    def download_file(self, drive_id: str) -> bytes:
        import io

        from googleapiclient.http import MediaIoBaseDownload

        buf = io.BytesIO()
        request = self._service.files().get_media(fileId=drive_id)
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buf.getvalue()


class ImageFetcher(Protocol):
    """The single choke point for image bytes (SPEC.md). Nothing else in the
    codebase may fetch image data."""

    def get(self, drive_id: str, size: int | Literal["orig"] = 512) -> Path: ...


class ThumbnailClient(Protocol):
    """Drive metadata/media operations the fetcher needs. Tests use a fake."""

    def get_thumbnail_link(self, drive_id: str) -> str | None: ...

    def download_file(self, drive_id: str) -> bytes: ...


class FetchError(RuntimeError):
    """Image bytes could not be fetched after retries."""


def _rewrite_size(link: str, size: int) -> str:
    """Rewrite the trailing =s{N} size suffix of a Drive thumbnail link."""
    if re.search(r"=s\d+(-c)?$", link):
        return re.sub(r"=s\d+(-c)?$", f"=s{size}", link)
    return f"{link}=s{size}"


class DriveImageFetcher:
    """Fetches size-parameterized thumbnails, caching to cache_dir.

    - Cache hit: return cache/{drive_id}_{size}.jpg if present.
    - Miss: fetch thumbnailLink with the size suffix rewritten. Thumbnail
      links expire — on 403/404 refresh the link via files.get, retry once.
    - No thumbnailLink: fetch original bytes and downscale locally (the
      only case where original bytes transit, and only as a fallback).
    - size="orig": fetch original bytes, cache as-is. Unused until Phase 7.
    - Exponential backoff on 429/5xx always; a first 403/404 is treated as
      an expired link (refresh via files.get, retry once), and 403s on the
      refreshed link are treated as rate limiting and backed off too.
    - Cache writes are atomic (temp file + rename) so an interrupted write
      can never poison a cache entry.
    - Every failure surfaces as FetchError; no client/transport exception
      escapes this abstraction.
    """

    MAX_ATTEMPTS = 4

    def __init__(
        self,
        db_path: Path | str,
        client: ThumbnailClient,
        session: Any,
        cache_dir: Path | str,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._db_path = db_path
        self._client = client
        self._session = session
        self._cache_dir = Path(cache_dir)
        self._sleep = sleep

    def get(self, drive_id: str, size: int | Literal["orig"] = 512) -> Path:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        if size == "orig":
            path = self._cache_dir / f"{drive_id}_orig"
            if not path.exists():
                self._write_atomic(path, self._download(drive_id))
            return path

        path = self._cache_dir / f"{drive_id}_{size}.jpg"
        if path.exists():
            return path

        link = self._stored_link(drive_id)
        refreshed = False
        if link is None:
            link = self._refresh_link(drive_id)
            refreshed = True
        if link is not None:
            resp = self._request(_rewrite_size(link, size))
            if resp.status_code in (403, 404) and not refreshed:
                link = self._refresh_link(drive_id)
                if link is not None:
                    # a 403 on a freshly refreshed link is rate limiting,
                    # not expiry — back off on it
                    resp = self._request(_rewrite_size(link, size), retry_403=True)
            if link is not None:
                if resp.status_code != 200:
                    raise FetchError(
                        f"thumbnail fetch for {drive_id} failed "
                        f"with HTTP {resp.status_code}"
                    )
                self._write_atomic(path, resp.content)
                self._record_thumb_path(drive_id, path)
                return path
        # no thumbnailLink at all (rare): fetch original bytes, downscale
        # locally — the only case where original bytes ever transit
        self._downscale_original(drive_id, size, path)
        self._record_thumb_path(drive_id, path)
        return path

    def _request(self, url: str, retry_403: bool = False) -> Any:
        """GET with exponential backoff on 429/5xx (and 403 when asked)."""
        resp = None
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                resp = self._session.get(url)
            except Exception as exc:
                raise FetchError(f"thumbnail request failed: {exc}") from exc
            retryable = (
                resp.status_code == 429
                or 500 <= resp.status_code < 600
                or (retry_403 and resp.status_code == 403)
            )
            if retryable and attempt < self.MAX_ATTEMPTS - 1:
                self._sleep(2**attempt)
                continue
            return resp
        return resp

    @staticmethod
    def _write_atomic(path: Path, data: bytes) -> None:
        """Write via temp file + rename: a crash mid-write can never leave a
        truncated file that path.exists() would treat as a cache hit."""
        import os
        import threading

        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.part")
        tmp.write_bytes(data)
        tmp.replace(path)

    def _download(self, drive_id: str) -> bytes:
        try:
            return self._client.download_file(drive_id)
        except Exception as exc:
            raise FetchError(f"download of {drive_id} failed: {exc}") from exc

    def _stored_link(self, drive_id: str) -> str | None:
        import sqlite3

        conn = sqlite3.connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT thumbnail_link FROM photos WHERE drive_id = ?", (drive_id,)
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def _refresh_link(self, drive_id: str) -> str | None:
        try:
            link = self._client.get_thumbnail_link(drive_id)
        except Exception as exc:
            raise FetchError(
                f"thumbnail-link refresh for {drive_id} failed: {exc}"
            ) from exc
        if link is not None:
            self._update_photo(drive_id, "thumbnail_link", link)
        return link

    def _record_thumb_path(self, drive_id: str, path: Path) -> None:
        self._update_photo(drive_id, "thumb_path", str(path))

    def _update_photo(self, drive_id: str, column: str, value: str) -> None:
        import sqlite3

        assert column in ("thumbnail_link", "thumb_path")
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                f"UPDATE photos SET {column} = ? WHERE drive_id = ?",
                (value, drive_id),
            )
            conn.commit()
        finally:
            conn.close()

    def _downscale_original(self, drive_id: str, size: int, path: Path) -> None:
        import io

        from PIL import Image

        data = self._download(drive_id)
        try:
            img = Image.open(io.BytesIO(data))
            img.thumbnail((size, size))
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "JPEG")
        except Exception as exc:
            raise FetchError(f"downscale of {drive_id} failed: {exc}") from exc
        self._write_atomic(path, buf.getvalue())
