"""Google Drive integration: OAuth, image listing.

Scope is drive.readonly — v1 never modifies anything in Drive.
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

log = logging.getLogger("doppel.fetch")

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
# Write scope used ONLY by the explicit, confirmed move-to-trash action. The
# scanning pipeline (sync, thumbnails, detection) always runs on the read-only
# scope above; nothing here ever deletes permanently — trashing is reversible.
DRIVE_WRITE_SCOPES = ["https://www.googleapis.com/auth/drive"]

LIST_QUERY = "mimeType contains 'image/' and trashed = false"
LIST_FIELDS = (
    "nextPageToken, files(id, name, mimeType, size, md5Checksum, "
    "imageMediaMetadata(width, height), createdTime, modifiedTime, "
    "thumbnailLink, parents)"
)
FOLDER_MIME = "application/vnd.google-apps.folder"
PAGE_SIZE = 1000

# how many parent-folder clauses to OR into one files.list query
PARENT_QUERY_CHUNK = 20

_FOLDER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,}$")


class DriveClient(Protocol):
    """The subset of the Drive API the sync job needs. Tests use a fake."""

    def list_images_page(
        self,
        page_token: str | None = None,
        parent_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """One page of files.list: {'files': [...], 'nextPageToken': str | absent}.

        parent_ids restricts results to direct children of those folders.
        """
        ...

    def list_folders_page(
        self, parent_id: str, page_token: str | None = None
    ) -> dict[str, Any]:
        """One page of subfolders directly under parent_id."""
        ...


def parse_folder_input(text: str) -> str | None:
    """Extract a Drive folder id from a pasted URL or raw id; None if the
    input is empty (= whole Drive) — raises ValueError if unparseable."""
    text = text.strip()
    if not text:
        return None
    match = re.search(r"folders/([A-Za-z0-9_-]+)", text)
    if match:
        return match.group(1)
    if _FOLDER_ID_RE.match(text):
        return text
    raise ValueError(
        "could not parse a folder id — paste the folder URL from Drive "
        "(…/drive/folders/<id>) or the id itself"
    )


def collect_folder_tree(client: DriveClient, root_id: str) -> list[str]:
    """The root folder id plus every descendant folder id (BFS). Drive has
    no recursive query, so the tree is walked one level at a time."""
    seen = {root_id}
    queue = [root_id]
    while queue:
        parent = queue.pop(0)
        page_token: str | None = None
        while True:
            page = client.list_folders_page(parent, page_token)
            for folder in page.get("files", []):
                if folder["id"] not in seen:
                    seen.add(folder["id"])
                    queue.append(folder["id"])
            page_token = page.get("nextPageToken")
            if not page_token:
                break
    return sorted(seen)


def web_auth_flow(
    credentials_path: Path | str,
    redirect_uri: str,
    scopes: list[str] | None = None,
) -> Any:
    """OAuth flow for the setup wizard: the consent redirect comes back to
    our own /oauth/callback route instead of a throwaway local server.
    Desktop-app OAuth clients accept any loopback redirect.

    Defaults to the read-only SCOPES. Pass DRIVE_WRITE_SCOPES for the separate
    move-to-trash sign-in, where the user authorizes as the file OWNER."""
    from google_auth_oauthlib.flow import Flow

    return Flow.from_client_secrets_file(
        str(credentials_path), scopes=scopes or SCOPES, redirect_uri=redirect_uri
    )


SERVICE_ACCOUNT_PATH = "service_account.json"


class CredentialsRequired(RuntimeError):
    """Interactive OAuth is needed but the caller cannot run it."""


class TrashNotAuthorized(RuntimeError):
    """Drive is connected read-only; moving files to Trash needs write access."""


def classify_trash_error(exc: Exception) -> tuple[str, str]:
    """Map a failed move-to-trash to a (code, human-readable reason) pair so the
    UI can group failures by cause instead of dumping raw Drive HttpError text.

    The dominant real-world cause is that the connected identity is not the
    file's OWNER: Google Drive only lets an owner set trashed=true, so a service
    account — or any non-owner, even one with edit access to a shared folder —
    gets 403 insufficientFilePermissions."""
    status = getattr(getattr(exc, "resp", None), "status", None)
    text = str(exc).lower()
    if (
        "insufficientfilepermissions" in text
        or "does not have sufficient permission" in text
        or "shared read-only" in text
        or status == 403
    ):
        return (
            "not_owner",
            "Not yours to move — Google Drive only lets a file's owner send it to "
            "Trash. The connected account isn't the owner (or reached the file "
            "through a read-only share).",
        )
    if status == 404 or "notfound" in text or "not found" in text:
        return ("missing", "Already gone from Drive — nothing left to move.")
    if status == 429 or "ratelimit" in text or "rate limit" in text:
        return ("rate_limited", "Drive rate-limited the request — try again shortly.")
    return ("other", "Drive refused the request.")


def load_service_account_credentials(
    path: Path | str = SERVICE_ACCOUNT_PATH,
    scopes: list[str] | None = None,
) -> Any:
    """Build Drive credentials from a service-account key file.

    Defaults to the read-only scope. Pass DRIVE_WRITE_SCOPES only for the
    move-to-trash action. No OAuth flow, consent screen, or token expiry —
    the app authenticates as the service account, which sees only the folders
    a user has shared with its email. Never log or print the key contents.
    """
    from google.oauth2 import service_account

    return service_account.Credentials.from_service_account_file(
        str(path), scopes=scopes or SCOPES
    )


def service_account_email(path: Path | str = SERVICE_ACCOUNT_PATH) -> str | None:
    """The client_email from a service-account key file — this is what the
    user shares their Drive folder with. Returns None if unreadable."""
    import json

    try:
        return json.loads(Path(path).read_text()).get("client_email")
    except (OSError, ValueError):
        return None


def is_service_account_key(data: bytes) -> bool:
    """True if the uploaded JSON is a service-account key (not an OAuth client)."""
    import json

    try:
        parsed = json.loads(data)
    except ValueError:
        return False
    return parsed.get("type") == "service_account" and "client_email" in parsed


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
                "Drive authorization required — open the /setup wizard and "
                "click “authorize with Google”."
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


TRASH_TOKEN_PATH = "token_write.json"


def load_trash_oauth_credentials(token_path: Path | str = TRASH_TOKEN_PATH) -> Any:
    """The write-scoped OAuth token used ONLY by move-to-trash. The user signs in
    as themselves — the file OWNER — so Google Drive permits trashed=true on their
    own files (a service account never can). Kept in a SEPARATE token file so it
    never disturbs the read-only scanning connection.

    Returns valid credentials, or None when the token is absent, corrupt, expired
    beyond refresh, or missing the write scope. Refreshes in place when possible.
    Never log or print the token contents."""
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    token_path = Path(token_path)
    if not token_path.exists():
        return None
    try:
        creds = Credentials.from_authorized_user_file(
            str(token_path), DRIVE_WRITE_SCOPES
        )
    except (ValueError, KeyError):
        return None
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            token_path.write_text(creds.to_json())
        except RefreshError:
            return None
    if not creds or not creds.valid:
        return None
    granted = set(getattr(creds, "scopes", None) or [])
    if not granted & set(DRIVE_WRITE_SCOPES):
        return None  # a read-only token can't trash — treat as not connected
    return creds


class GoogleDriveClient:
    """Real Drive client. Everything network-facing lives behind this."""

    def __init__(self, credentials: Any) -> None:
        # httplib2, the transport under googleapiclient, is NOT thread-safe: two
        # threads sharing one service object interleave on the same TLS socket
        # and corrupt the stream. That surfaces as WRONG_VERSION_NUMBER, garbled
        # response bytes (an unreadable image), or a hard OpenSSL segfault that
        # kills the whole process. The deep scans fan fetches out across a thread
        # pool, so each thread lazily builds and reuses its OWN service from the
        # shared (thread-safe) credentials.
        self._credentials = credentials
        self._local = threading.local()

    # httplib2's default socket has NO timeout, so a stalled Drive connection
    # (a routine thumbnail-link refresh, or the original-bytes download fallback)
    # would hang a worker thread indefinitely — and a hung worker silently wedges
    # the whole stage, since the pool never sees it finish. Bound every socket at
    # this many seconds so a stuck call fails fast; the fetcher turns that failure
    # into a FetchError the stage records and skips.
    _TIMEOUT_S = 30

    @property
    def _service(self) -> Any:
        service = getattr(self._local, "service", None)
        if service is None:
            import google_auth_httplib2
            import httplib2
            from googleapiclient.discovery import build

            # a per-thread AuthorizedHttp over a timeout-bounded httplib2 socket;
            # http= carries the auth, so credentials= must not also be passed.
            authed_http = google_auth_httplib2.AuthorizedHttp(
                self._credentials, http=httplib2.Http(timeout=self._TIMEOUT_S)
            )
            service = build("drive", "v3", http=authed_http, cache_discovery=False)
            self._local.service = service
        return service

    def list_images_page(
        self,
        page_token: str | None = None,
        parent_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        query = LIST_QUERY
        if parent_ids:
            parents = " or ".join(
                f"'{self._sanitize_id(fid)}' in parents" for fid in parent_ids
            )
            query = f"({parents}) and {LIST_QUERY}"
        return (
            self._service.files()
            .list(
                q=query,
                pageSize=PAGE_SIZE,
                fields=LIST_FIELDS,
                pageToken=page_token,
                # include shared drives so scoping to a shared-drive folder works
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )

    def list_folders_page(
        self, parent_id: str, page_token: str | None = None
    ) -> dict[str, Any]:
        query = (
            f"'{self._sanitize_id(parent_id)}' in parents "
            f"and mimeType = '{FOLDER_MIME}' and trashed = false"
        )
        return (
            self._service.files()
            .list(
                q=query,
                pageSize=PAGE_SIZE,
                fields="nextPageToken, files(id, name)",
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )

    def get_folder(self, folder_id: str) -> dict[str, Any]:
        """Folder metadata (incl. parents for up-navigation); raises if the
        id is not a reachable folder. 'root' resolves to My Drive."""
        meta = (
            self._service.files()
            .get(
                fileId=folder_id,
                fields="id, name, mimeType, parents",
                supportsAllDrives=True,
            )
            .execute()
        )
        if meta.get("mimeType") != FOLDER_MIME:
            raise ValueError(f"{meta.get('name', folder_id)!r} is not a folder")
        return meta

    def trash_file(self, drive_id: str) -> None:
        """Move a file to Google Drive Trash — reversible, recoverable for 30
        days. This sets trashed=true; it NEVER permanently deletes. Requires a
        write-scoped credential (see DRIVE_WRITE_SCOPES) and edit access to the
        file; Drive raises otherwise (e.g. a folder shared read-only)."""
        self._service.files().update(
            fileId=self._sanitize_id(drive_id),
            body={"trashed": True},
            supportsAllDrives=True,
        ).execute()

    def list_child_folders(self, parent_id: str) -> list[dict[str, Any]]:
        """Every subfolder directly under parent_id (paginated to completion)."""
        folders: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            page = self.list_folders_page(parent_id, page_token)
            folders.extend(page.get("files", []))
            page_token = page.get("nextPageToken")
            if not page_token:
                break
        return folders

    def list_shared_folders(self) -> list[dict[str, Any]]:
        """Folders shared with this account — the entry point in service-account
        mode, where the account's own My Drive is empty and the user grants
        access by sharing folders with the service-account email."""
        query = (
            f"sharedWithMe = true and mimeType = '{FOLDER_MIME}' and trashed = false"
        )
        folders: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            page = (
                self._service.files()
                .list(
                    q=query,
                    pageSize=PAGE_SIZE,
                    fields="nextPageToken, files(id, name)",
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            folders.extend(page.get("files", []))
            page_token = page.get("nextPageToken")
            if not page_token:
                break
        return folders

    @staticmethod
    def _sanitize_id(drive_id: str) -> str:
        # 'root' is Drive's alias for My Drive's top folder
        if drive_id == "root" or _FOLDER_ID_RE.match(drive_id):
            return drive_id
        raise ValueError(f"invalid Drive id {drive_id!r}")

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

    # --- write ops (WebP->PNG conversion) -------------------------------------
    # These CREATE and MOVE files, so they need a write-scoped OWNER credential
    # (a service account is not the owner and would 403). They never delete.

    def root_folder_id(self) -> str:
        """The id of this account's My Drive root, for re-parenting files that
        live at the top level."""
        return (
            self._service.files()
            .get(fileId="root", fields="id", supportsAllDrives=True)
            .execute()["id"]
        )

    def ensure_child_folder(self, name: str, parent_id: str | None) -> str:
        """Find, or create, a subfolder `name` directly under parent_id (or the
        root when parent_id is None); return its id. Idempotent."""
        safe = name.replace("\\", "\\\\").replace("'", "\\'")
        query = f"name = '{safe}' and mimeType = '{FOLDER_MIME}' and trashed = false"
        if parent_id:
            query += f" and '{self._sanitize_id(parent_id)}' in parents"
        page = (
            self._service.files()
            .list(
                q=query,
                fields="files(id, name)",
                pageSize=1,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        hits = page.get("files", [])
        if hits:
            return hits[0]["id"]
        body: dict[str, Any] = {"name": name, "mimeType": FOLDER_MIME}
        if parent_id:
            body["parents"] = [self._sanitize_id(parent_id)]
        created = (
            self._service.files()
            .create(body=body, fields="id", supportsAllDrives=True)
            .execute()
        )
        return created["id"]

    def find_child(self, name: str, parent_id: str | None) -> str | None:
        """The id of a non-trashed file named `name` under parent_id, if one
        exists — so the converter can avoid creating a duplicate PNG."""
        safe = name.replace("\\", "\\\\").replace("'", "\\'")
        query = f"name = '{safe}' and trashed = false"
        if parent_id:
            query += f" and '{self._sanitize_id(parent_id)}' in parents"
        page = (
            self._service.files()
            .list(
                q=query,
                fields="files(id)",
                pageSize=1,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        hits = page.get("files", [])
        return hits[0]["id"] if hits else None

    def upload_file(
        self, name: str, parent_id: str | None, data: bytes, mime_type: str
    ) -> str:
        """Create a new file with `data` under parent_id (or root); return its id."""
        import io

        from googleapiclient.http import MediaIoBaseUpload

        body: dict[str, Any] = {"name": name}
        if parent_id:
            body["parents"] = [self._sanitize_id(parent_id)]
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime_type, resumable=True)
        created = (
            self._service.files()
            .create(body=body, media_body=media, fields="id", supportsAllDrives=True)
            .execute()
        )
        return created["id"]

    def move_file(self, drive_id: str, add_parent: str, remove_parent: str) -> None:
        """Re-parent a file (add add_parent, remove remove_parent) — used to set a
        converted WebP original aside in the trash folder. Reversible; NEVER
        deletes."""
        self._service.files().update(
            fileId=self._sanitize_id(drive_id),
            addParents=self._sanitize_id(add_parent),
            removeParents=self._sanitize_id(remove_parent),
            fields="id, parents",
            supportsAllDrives=True,
        ).execute()


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
    # (connect, read) timeouts in seconds: a hung socket must never wedge a
    # worker thread for the rest of the scan. Generous, since the thumbnail CDN
    # is normally fast — this only trips on a genuinely stuck connection.
    TIMEOUT = (10, 30)

    def __init__(
        self,
        db_path: Path | str,
        client: ThumbnailClient,
        session_factory: Callable[[], Any],
        cache_dir: Path | str,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._db_path = db_path
        self._client = client
        # requests/AuthorizedSession is not thread-safe. The deep scans fetch
        # from a thread pool, so each worker thread lazily builds and then
        # reuses its OWN session (per-thread keep-alive is also a big speed win:
        # one TLS handshake per thread, then warm-connection reuse, instead of a
        # fresh handshake per image).
        self._session_factory = session_factory
        self._local = threading.local()
        self._cache_dir = Path(cache_dir)
        self._sleep = sleep

    @property
    def _session(self) -> Any:
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._session_factory()
            self._local.session = session
        return session

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
                # a 200 is not proof of an image: the CDN can hand back an HTML
                # interstitial, and a torn keep-alive read or truncated body can
                # slip through. Decoding here keeps a non-image OUT of the cache —
                # otherwise it becomes a permanent cache hit that crashes the
                # embed stage (which decodes on the main thread, outside the
                # per-item guard) on every resume, wedging the scan for good.
                self._ensure_decodable(drive_id, resp.content)
                self._write_atomic(path, resp.content)
                self._record_thumb_path(drive_id, path)
                return path
        # no thumbnailLink at all (rare): fetch original bytes, downscale
        # locally — the only case where original bytes ever transit
        self._downscale_original(drive_id, size, path)
        self._record_thumb_path(drive_id, path)
        return path

    @staticmethod
    def _ensure_decodable(drive_id: str, data: bytes) -> None:
        """Raise FetchError unless ``data`` fully decodes as an image. verify()
        alone misses truncation, so force a full decode with load()."""
        import io

        from PIL import Image

        try:
            with Image.open(io.BytesIO(data)) as img:
                img.load()
        except Exception as exc:  # noqa: BLE001 — any decode failure = not an image
            raise FetchError(
                f"thumbnail for {drive_id} returned HTTP 200 but the body was "
                f"not a decodable image ({type(exc).__name__})"
            ) from exc

    def _backoff(self, attempt: int, reason: str) -> None:
        """Sleep 2**attempt seconds plus up to 0.5s of jitter. The jitter
        de-syncs the many worker threads (especially under Boost) so they don't
        all retry a rate-limited CDN in lockstep. Logged so a 429 burst is
        visible in logs/ instead of looking like a stall."""
        delay = 2**attempt + random.uniform(0, 0.5)
        log.info("fetch backoff %.1fs (retry %d): %s", delay, attempt + 1, reason)
        self._sleep(delay)

    def _request(self, url: str, retry_403: bool = False) -> Any:
        """GET with exponential backoff on 429/5xx (and 403 when asked), and on
        transient transport errors (reset connection, read timeout). With a
        per-thread session those errors are genuinely transient rather than
        shared-state corruption, so a short retry recovers instead of dropping
        the photo — only a persistent failure surfaces as FetchError."""
        resp = None
        for attempt in range(self.MAX_ATTEMPTS):
            last = attempt == self.MAX_ATTEMPTS - 1
            try:
                resp = self._session.get(url, timeout=self.TIMEOUT)
            except Exception as exc:  # noqa: BLE001 — retried, then surfaced as FetchError
                if last:
                    raise FetchError(f"thumbnail request failed: {exc}") from exc
                self._backoff(attempt, str(exc))
                continue
            retryable = (
                resp.status_code == 429
                or 500 <= resp.status_code < 600
                or (retry_403 and resp.status_code == 403)
            )
            if retryable and not last:
                self._backoff(attempt, f"HTTP {resp.status_code}")
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
        # sqlite3.connect's default timeout=5.0 sets busy_timeout=5000ms, so this
        # short-lived write waits (not errors) if a scan's writer holds the WAL
        # lock — concurrent fetch-during-scan is safe without extra pragmas.
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
