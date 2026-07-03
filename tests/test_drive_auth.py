"""get_credentials recovery paths. All google libs are monkeypatched — no network."""

from __future__ import annotations

import json
from typing import Any

import google.oauth2.credentials
import google_auth_oauthlib.flow
import pytest
from google.auth.exceptions import RefreshError

from doppel.drive import get_credentials


class FakeCreds:
    def __init__(
        self, valid: bool, expired: bool = False, refresh_token: str | None = None
    ) -> None:
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self.refreshed = False

    def refresh(self, request: Any) -> None:
        raise RefreshError("invalid_grant: Token has been expired or revoked")

    def to_json(self) -> str:
        return json.dumps({"fake": "token"})


class FakeFlow:
    def __init__(self, creds: FakeCreds) -> None:
        self._creds = creds

    def run_local_server(self, **kwargs: Any) -> FakeCreds:
        return self._creds


@pytest.fixture
def paths(tmp_path):
    credentials = tmp_path / "credentials.json"
    credentials.write_text("{}")
    token = tmp_path / "token.json"
    return credentials, token


def _patch_flow(monkeypatch, fresh: FakeCreds) -> None:
    monkeypatch.setattr(
        google_auth_oauthlib.flow.InstalledAppFlow,
        "from_client_secrets_file",
        classmethod(lambda cls, *a, **k: FakeFlow(fresh)),
    )


def test_revoked_refresh_token_falls_back_to_interactive_flow(
    monkeypatch, paths
) -> None:
    credentials, token = paths
    token.write_text(json.dumps({"stale": True}))
    stale = FakeCreds(valid=False, expired=True, refresh_token="stale-token")
    monkeypatch.setattr(
        google.oauth2.credentials.Credentials,
        "from_authorized_user_file",
        classmethod(lambda cls, *a, **k: stale),
    )
    fresh = FakeCreds(valid=True)
    _patch_flow(monkeypatch, fresh)

    creds = get_credentials(credentials, token)

    assert creds is fresh
    assert json.loads(token.read_text()) == {"fake": "token"}


def test_corrupt_token_file_falls_back_to_interactive_flow(monkeypatch, paths) -> None:
    credentials, token = paths
    token.write_text("not json at all")
    monkeypatch.setattr(
        google.oauth2.credentials.Credentials,
        "from_authorized_user_file",
        classmethod(
            lambda cls, *a, **k: (_ for _ in ()).throw(ValueError("bad token"))
        ),
    )
    fresh = FakeCreds(valid=True)
    _patch_flow(monkeypatch, fresh)

    creds = get_credentials(credentials, token)

    assert creds is fresh


def test_missing_credentials_file_raises_helpful_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="credentials.json"):
        get_credentials(tmp_path / "credentials.json", tmp_path / "token.json")
