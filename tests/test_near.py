import pytest

import doppel.stages.near as near_mod
from doppel.stages.near import compute_hashes, hamming_hex, run_near
from tests.fakes import FakeImageFetcher, insert_photo
from tests.images import as_jpeg, color_variant, structured_image, unrelated_image


@pytest.fixture(scope="module")
def images() -> dict[str, bytes]:
    base = structured_image()
    return {
        "base": as_jpeg(base),
        "resized": as_jpeg(base.resize((400, 300))),
        "recompressed": as_jpeg(base, quality=25),
        "variant": as_jpeg(color_variant(base)),
        "other": as_jpeg(unrelated_image()),
    }


def make_fetcher(config, images, subset=None):
    picked = {k: images[k] for k in (subset or images)}
    return FakeImageFetcher(config.cache_dir, images=picked)


def near_groups(conn) -> list[dict]:
    groups = []
    for g in conn.execute("SELECT * FROM groups WHERE tier = 'near' ORDER BY id"):
        members = {
            row["drive_id"]
            for row in conn.execute(
                """
                SELECT p.drive_id FROM group_members m
                JOIN photos p ON p.id = m.photo_id WHERE m.group_id = ?
                """,
                (g["id"],),
            )
        }
        groups.append(
            {"id": g["id"], "members": members, "color_variant": g["color_variant"]}
        )
    return groups


def seed(conn, images, names) -> None:
    for name in names:
        insert_photo(conn, name, name=f"{name}.jpg", md5=f"md5-{name}")


def test_near_groups_copies_but_not_unrelated(conn, config, images) -> None:
    seed(conn, images, ["base", "resized", "recompressed", "other"])
    fetcher = make_fetcher(config, images)

    run_near(conn, fetcher, config)

    groups = near_groups(conn)
    assert len(groups) == 1
    assert groups[0]["members"] == {"base", "resized", "recompressed"}
    assert groups[0]["color_variant"] == 0


def test_near_flags_color_variants(conn, config, images) -> None:
    seed(conn, images, ["base", "variant"])
    fetcher = make_fetcher(config, images)

    run_near(conn, fetcher, config)

    groups = near_groups(conn)
    assert len(groups) == 1
    assert groups[0]["members"] == {"base", "variant"}
    assert groups[0]["color_variant"] == 1


def test_near_skips_byte_identical_pairs(conn, config, images) -> None:
    # same md5 == byte-identical: exact tier's job, not near's
    insert_photo(conn, "base", md5="same")
    insert_photo(conn, "resized", md5="same")
    fetcher = make_fetcher(config, images)

    run_near(conn, fetcher, config)

    assert near_groups(conn) == []


def test_near_scores_are_hamming_vs_anchor(conn, config, images) -> None:
    seed(conn, images, ["base", "resized"])
    fetcher = make_fetcher(config, images)

    run_near(conn, fetcher, config)

    rows = conn.execute(
        """
        SELECT p.drive_id, p.phash, m.score FROM group_members m
        JOIN photos p ON p.id = m.photo_id ORDER BY p.id
        """
    ).fetchall()
    anchor = rows[0]
    for row in rows:
        assert row["score"] == hamming_hex(row["phash"], anchor["phash"])


def test_near_resumes_without_rehashing(conn, config, images, monkeypatch) -> None:
    seed(conn, images, ["base", "resized", "other"])

    class Interrupt(Exception):
        pass

    fetcher = make_fetcher(config, images)
    fetcher.images["other"] = Interrupt("fetch blew up")  # third photo fails
    run_near(conn, fetcher, config)  # completes; 'other' left unhashed

    hashed = {
        row["drive_id"]
        for row in conn.execute("SELECT drive_id FROM photos WHERE phash IS NOT NULL")
    }
    assert hashed == {"base", "resized"}

    calls: list[str] = []
    real_compute = compute_hashes
    monkeypatch.setattr(
        near_mod,
        "compute_hashes",
        lambda img: calls.append("hash") or real_compute(img),
    )
    run_near(conn, make_fetcher(config, images), config)

    assert calls == ["hash"]  # only 'other' needed hashing on the second run
    assert len(near_groups(conn)) == 1


def test_near_interrupt_marks_scan_failed(conn, config, images) -> None:
    seed(conn, images, ["base"])
    fetcher = make_fetcher(config, images)
    fetcher.images["base"] = KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        run_near(conn, fetcher, config)

    scan = conn.execute("SELECT * FROM scans ORDER BY id DESC LIMIT 1").fetchone()
    assert scan["status"] == "failed"


def test_near_rebuild_is_idempotent(conn, config, images) -> None:
    seed(conn, images, ["base", "resized"])

    run_near(conn, make_fetcher(config, images), config)
    run_near(conn, make_fetcher(config, images), config)

    assert len(near_groups(conn)) == 1
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM group_members WHERE group_id IN "
        "(SELECT id FROM groups WHERE tier = 'near')"
    ).fetchone()["n"]
    assert n == 2


def test_near_ignores_missing_photos(conn, config, images) -> None:
    insert_photo(conn, "base", md5="a")
    insert_photo(conn, "resized", md5="b", status="missing")
    fetcher = make_fetcher(config, images)

    run_near(conn, fetcher, config)

    assert near_groups(conn) == []
    assert ("resized", config.thumb_size) not in fetcher.calls
