"""Google Drive integration: OAuth, image listing.

Scope is drive.readonly — v1 never modifies anything in Drive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

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


def get_credentials(
    credentials_path: Path | str = "credentials.json",
    token_path: Path | str = "token.json",
) -> Any:
    """Run the OAuth installed-app flow, caching the refresh token.

    Never log or print the credential contents.
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
