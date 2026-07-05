import io

import pytest
from PIL import Image

from doppel.db import connect
from doppel.drive import DriveImageFetcher, FetchError, _rewrite_size
from tests.fakes import (
    FakeResponse,
    FakeSession,
    FakeThumbClient,
    insert_photo,
    jpeg_bytes,
)


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    conn = connect(path)
    insert_photo(conn, "pic1", thumbnail_link="https://lh3.example/abc=s220")
    conn.close()
    return path


def make_fetcher(db_path, tmp_path, client, session):
    sleeps: list[float] = []
    fetcher = DriveImageFetcher(
        db_path=db_path,
        client=client,
        session_factory=lambda: session,
        cache_dir=tmp_path / "cache",
        sleep=sleeps.append,
    )
    return fetcher, sleeps


def test_rewrite_size() -> None:
    assert _rewrite_size("https://x/abc=s220", 512) == "https://x/abc=s512"
    assert _rewrite_size("https://x/abc=s220-c", 512) == "https://x/abc=s512"
    assert _rewrite_size("https://x/abc", 512) == "https://x/abc=s512"


def test_cache_hit_never_touches_network(db_path, tmp_path) -> None:
    session = FakeSession([])
    fetcher, _ = make_fetcher(db_path, tmp_path, FakeThumbClient(), session)
    cached = tmp_path / "cache" / "pic1_512.jpg"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"cached")

    path = fetcher.get("pic1", 512)

    assert path == cached
    assert session.requests == []


def test_fetch_rewrites_size_and_records_thumb_path(db_path, tmp_path) -> None:
    body = jpeg_bytes()
    session = FakeSession([FakeResponse(200, body)])
    fetcher, _ = make_fetcher(db_path, tmp_path, FakeThumbClient(), session)

    path = fetcher.get("pic1", 512)

    assert session.requests == ["https://lh3.example/abc=s512"]
    assert path.read_bytes() == body
    conn = connect(db_path)
    row = conn.execute(
        "SELECT thumb_path FROM photos WHERE drive_id = 'pic1'"
    ).fetchone()
    assert row["thumb_path"] == str(path)


def test_expired_link_refreshes_and_retries_once(db_path, tmp_path) -> None:
    body = jpeg_bytes()
    session = FakeSession([FakeResponse(404), FakeResponse(200, body)])
    client = FakeThumbClient(link="https://lh3.example/new=s220")
    fetcher, _ = make_fetcher(db_path, tmp_path, client, session)

    path = fetcher.get("pic1", 512)

    assert client.link_requests == ["pic1"]
    assert session.requests == [
        "https://lh3.example/abc=s512",
        "https://lh3.example/new=s512",
    ]
    assert path.read_bytes() == body
    conn = connect(db_path)
    row = conn.execute(
        "SELECT thumbnail_link FROM photos WHERE drive_id = 'pic1'"
    ).fetchone()
    assert row["thumbnail_link"] == "https://lh3.example/new=s220"


def test_no_thumbnail_link_falls_back_to_downscaled_original(tmp_path) -> None:
    db = tmp_path / "test.db"
    conn = connect(db)
    insert_photo(conn, "raw1", thumbnail_link=None)
    conn.close()
    client = FakeThumbClient(link=None, file_bytes=jpeg_bytes(size=(1024, 768)))
    fetcher, _ = make_fetcher(db, tmp_path, client, FakeSession([]))

    path = fetcher.get("raw1", 512)

    assert client.download_requests == ["raw1"]
    img = Image.open(io.BytesIO(path.read_bytes()))
    assert img.format == "JPEG"
    assert max(img.size) <= 512


def test_backoff_on_transient_errors(db_path, tmp_path) -> None:
    body = jpeg_bytes()
    session = FakeSession(
        [FakeResponse(500), FakeResponse(429), FakeResponse(200, body)]
    )
    fetcher, sleeps = make_fetcher(db_path, tmp_path, FakeThumbClient(), session)

    path = fetcher.get("pic1", 512)

    assert path.read_bytes() == body
    # exponential backoff (1s, 2s) plus up to 0.5s of anti-lockstep jitter
    assert [int(s) for s in sleeps] == [1, 2]
    assert all(0 <= s - int(s) < 0.5 for s in sleeps)


def test_persistent_failure_raises_fetch_error(db_path, tmp_path) -> None:
    session = FakeSession([FakeResponse(500)] * DriveImageFetcher.MAX_ATTEMPTS)
    fetcher, _ = make_fetcher(db_path, tmp_path, FakeThumbClient(), session)

    with pytest.raises(FetchError):
        fetcher.get("pic1", 512)


def test_orig_fetch_caches_as_is(tmp_path) -> None:
    db = tmp_path / "test.db"
    conn = connect(db)
    insert_photo(conn, "o1")
    conn.close()
    client = FakeThumbClient(file_bytes=b"original-bytes")
    fetcher, _ = make_fetcher(db, tmp_path, client, FakeSession([]))

    path = fetcher.get("o1", "orig")
    again = fetcher.get("o1", "orig")

    assert path == again
    assert path.read_bytes() == b"original-bytes"
    assert client.download_requests == ["o1"]  # cached on second call


def test_403_on_refreshed_link_backs_off(db_path, tmp_path) -> None:
    # first 403 = expired link; 403s on the refreshed link = rate limiting
    responses = [FakeResponse(403)] + [
        FakeResponse(403)
    ] * DriveImageFetcher.MAX_ATTEMPTS
    session = FakeSession(responses)
    client = FakeThumbClient(link="https://lh3.example/new=s220")
    fetcher, sleeps = make_fetcher(db_path, tmp_path, client, session)

    with pytest.raises(FetchError):
        fetcher.get("pic1", 512)

    assert client.link_requests == ["pic1"]  # exactly one refresh
    assert [int(s) for s in sleeps] == [1, 2, 4]  # backoff (+jitter), none after last


def test_no_sleep_after_final_attempt(db_path, tmp_path) -> None:
    session = FakeSession([FakeResponse(500)] * DriveImageFetcher.MAX_ATTEMPTS)
    fetcher, sleeps = make_fetcher(db_path, tmp_path, FakeThumbClient(), session)

    with pytest.raises(FetchError):
        fetcher.get("pic1", 512)

    assert [int(s) for s in sleeps] == [1, 2, 4]  # not [1, 2, 4, 8]


def test_stale_link_with_no_replacement_falls_back_to_original(
    db_path, tmp_path
) -> None:
    # link 404s and files.get says the file no longer has a thumbnailLink
    session = FakeSession([FakeResponse(404)])
    client = FakeThumbClient(link=None, file_bytes=jpeg_bytes(size=(1024, 768)))
    fetcher, _ = make_fetcher(db_path, tmp_path, client, session)

    path = fetcher.get("pic1", 512)

    assert client.download_requests == ["pic1"]
    assert path.exists()


def test_client_errors_surface_as_fetch_error(db_path, tmp_path) -> None:
    class ExplodingClient:
        def get_thumbnail_link(self, drive_id):
            raise RuntimeError("HttpError 404")

        def download_file(self, drive_id):
            raise RuntimeError("HttpError 404")

    session = FakeSession([FakeResponse(404)])
    fetcher, _ = make_fetcher(db_path, tmp_path, ExplodingClient(), session)

    with pytest.raises(FetchError):
        fetcher.get("pic1", 512)


def test_no_partial_cache_files_left_behind(db_path, tmp_path) -> None:
    session = FakeSession([FakeResponse(200, jpeg_bytes())])
    fetcher, _ = make_fetcher(db_path, tmp_path, FakeThumbClient(), session)

    fetcher.get("pic1", 512)

    leftovers = list((tmp_path / "cache").glob("*.part"))
    assert leftovers == []


def test_non_image_200_body_is_not_cached(db_path, tmp_path) -> None:
    """A 200 whose body is not a decodable image (CDN error page, truncated
    body) must raise instead of caching poison — otherwise it becomes a
    permanent cache hit that crashes the decode stage on every resume."""
    session = FakeSession([FakeResponse(200, b"<html>rate limited</html>")])
    fetcher, _ = make_fetcher(db_path, tmp_path, FakeThumbClient(), session)

    with pytest.raises(FetchError):
        fetcher.get("pic1", 512)

    # nothing poisonous (and no temp .part) left behind — the item can retry
    assert not (tmp_path / "cache" / "pic1_512.jpg").exists()
    assert list((tmp_path / "cache").glob("*.part")) == []


def test_client_service_is_built_with_a_socket_timeout(monkeypatch) -> None:
    """The per-thread Drive service must sit on a timeout-bounded httplib2
    socket; without it a stalled get_thumbnail_link/download_file would hang a
    worker (and wedge the whole stage) forever."""
    import google_auth_httplib2
    import googleapiclient.discovery
    import httplib2

    from doppel.drive import GoogleDriveClient

    captured: dict[str, object] = {}

    class FakeHttp:
        def __init__(self, timeout=None) -> None:
            captured["timeout"] = timeout

    monkeypatch.setattr(httplib2, "Http", FakeHttp)
    monkeypatch.setattr(
        google_auth_httplib2, "AuthorizedHttp", lambda creds, http=None: http
    )
    monkeypatch.setattr(
        googleapiclient.discovery, "build", lambda *a, **k: ("service", k.get("http"))
    )

    service = GoogleDriveClient(credentials="dummy")._service

    assert captured["timeout"] == GoogleDriveClient._TIMEOUT_S == 30
    assert isinstance(service[1], FakeHttp)  # service built over the bounded http


def test_session_is_per_thread(db_path, tmp_path) -> None:
    """Each worker thread must get its OWN session — sharing one requests/
    AuthorizedSession across the fetch thread pool corrupts the TLS stream and
    crashed real scans (WRONG_VERSION_NUMBER / unreadable image / segfault).
    Within a thread the session is reused (per-thread keep-alive)."""
    import threading

    built: list[int] = []
    build_lock = threading.Lock()

    def factory() -> FakeSession:
        with build_lock:
            built.append(threading.get_ident())
        return FakeSession([])

    fetcher = DriveImageFetcher(
        db_path=db_path,
        client=FakeThumbClient(),
        session_factory=factory,
        cache_dir=tmp_path / "cache",
    )

    # hold a strong ref to each thread's session so none is GC'd mid-test —
    # otherwise id() could be reused across threads and give a false collision
    per_thread_session: dict[int, FakeSession] = {}
    start = threading.Barrier(4)  # release all threads at once to force overlap

    def grab() -> None:
        start.wait()
        first = fetcher._session
        assert fetcher._session is first  # reused within the same thread
        per_thread_session[threading.get_ident()] = first

    threads = [threading.Thread(target=grab) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    sessions = list(per_thread_session.values())
    assert len(per_thread_session) == 4  # four distinct threads
    assert len({id(s) for s in sessions}) == 4  # each got its own session object
    assert sorted(built) == sorted(per_thread_session)  # built once per thread


def test_transient_transport_error_is_retried(db_path, tmp_path) -> None:
    """A dropped connection / read timeout on a per-thread session is transient,
    so the fetcher backs off and retries instead of dropping the photo."""

    class FlakySession:
        def __init__(self, fail_times: int, then: FakeResponse) -> None:
            self.fail_times = fail_times
            self.then = then
            self.calls = 0

        def get(self, url: str, timeout: object = None) -> FakeResponse:
            self.calls += 1
            if self.calls <= self.fail_times:
                raise ConnectionError("connection reset by peer")
            return self.then

    session = FlakySession(2, FakeResponse(200, jpeg_bytes()))
    fetcher, sleeps = make_fetcher(db_path, tmp_path, FakeThumbClient(), session)

    path = fetcher.get("pic1", 512)

    assert path.exists()
    assert session.calls == 3  # two failures, then success
    assert [int(s) for s in sleeps] == [1, 2]  # backoff (1s, 2s) + jitter
