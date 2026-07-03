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
        session=session,
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
    session = FakeSession([FakeResponse(200, b"jpegdata")])
    fetcher, _ = make_fetcher(db_path, tmp_path, FakeThumbClient(), session)

    path = fetcher.get("pic1", 512)

    assert session.requests == ["https://lh3.example/abc=s512"]
    assert path.read_bytes() == b"jpegdata"
    conn = connect(db_path)
    row = conn.execute(
        "SELECT thumb_path FROM photos WHERE drive_id = 'pic1'"
    ).fetchone()
    assert row["thumb_path"] == str(path)


def test_expired_link_refreshes_and_retries_once(db_path, tmp_path) -> None:
    session = FakeSession([FakeResponse(404), FakeResponse(200, b"fresh")])
    client = FakeThumbClient(link="https://lh3.example/new=s220")
    fetcher, _ = make_fetcher(db_path, tmp_path, client, session)

    path = fetcher.get("pic1", 512)

    assert client.link_requests == ["pic1"]
    assert session.requests == [
        "https://lh3.example/abc=s512",
        "https://lh3.example/new=s512",
    ]
    assert path.read_bytes() == b"fresh"
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
    session = FakeSession(
        [FakeResponse(500), FakeResponse(429), FakeResponse(200, b"ok")]
    )
    fetcher, sleeps = make_fetcher(db_path, tmp_path, FakeThumbClient(), session)

    path = fetcher.get("pic1", 512)

    assert path.read_bytes() == b"ok"
    assert sleeps == [1, 2]


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
