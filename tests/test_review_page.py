"""One-page continuous-scroll review: inline save, Keep Group, auto-resolve."""

import pytest
from fastapi.testclient import TestClient

from doppel.app import REVIEW_BATCH, create_app
from doppel.db import connect
from doppel.jobs import now
from tests.fakes import FakeImageFetcher, insert_photo


@pytest.fixture
def client(config):
    app = create_app(
        config=config, fetcher_factory=lambda cfg: FakeImageFetcher(cfg.cache_dir)
    )
    with TestClient(app) as c:
        c.app = app
        yield c


def make_exact_group(conn, key, sizes) -> int:
    """A group of photos sharing md5=key with the given sizes; returns group id."""
    cur = conn.execute(
        "INSERT INTO groups (tier, created_at) VALUES ('exact', ?)", (now(),)
    )
    gid = cur.lastrowid
    for i, size in enumerate(sizes):
        pid = insert_photo(
            conn, f"{key}-{i}", name=f"{key}-{i}.jpg", md5=key, size=size
        )
        conn.execute(
            "INSERT INTO group_members (group_id, photo_id) VALUES (?, ?)", (gid, pid)
        )
    conn.commit()
    return gid


def photo_ids(conn, gid):
    return [
        r["photo_id"]
        for r in conn.execute(
            "SELECT photo_id FROM group_members WHERE group_id = ? ORDER BY photo_id",
            (gid,),
        )
    ]


def test_review_page_renders_all_groups_inline(client, config):
    conn = connect(config.db_path)
    make_exact_group(conn, "a", [2000, 500])
    make_exact_group(conn, "b", [3000, 100])
    conn.close()

    page = client.get("/review", params={"tier": "exact"})
    assert page.status_code == 200
    # both groups' photos appear on the one page (no pagination/detail hop)
    assert "a-0.jpg" in page.text and "b-0.jpg" in page.text
    # largest is preselected keep, the rest trash
    assert "checked> keep" in page.text
    assert "checked> trash" in page.text


def test_infinite_scroll_batches(client, config):
    conn = connect(config.db_path)
    for i in range(REVIEW_BATCH + 3):  # more than one batch
        make_exact_group(conn, f"g{i}", [100 + i, 10])
    conn.close()

    first = client.get("/review", params={"tier": "exact"})
    # a sentinel requests the next page when revealed
    assert 'hx-get="/review/groups?tier=exact' in first.text
    assert "page=2" in first.text

    second = client.get("/review/groups", params={"tier": "exact", "page": 2})
    assert second.status_code == 200
    assert "review-group" in second.text  # more cards
    assert "page=3" not in second.text  # only 3 groups on page 2, no further batch


def test_inline_save_returns_updated_card(client, config):
    conn = connect(config.db_path)
    gid = make_exact_group(conn, "a", [2000, 500])
    big, small = photo_ids(conn, gid)
    conn.close()

    # htmx request (has HX-Request header) gets the card partial back, not a redirect
    resp = client.post(
        f"/groups/{gid}/decisions",
        data={f"action_{big}": "trash", f"action_{small}": "keep"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    assert f'id="group-{gid}"' in resp.text  # the swapped-in card
    conn = connect(config.db_path)
    actions = {
        r["photo_id"]: r["action"] for r in conn.execute("SELECT * FROM decisions")
    }
    conn.close()
    assert actions == {big: "trash", small: "keep"}


def test_keep_group_marks_all_keep(client, config):
    conn = connect(config.db_path)
    gid = make_exact_group(conn, "a", [2000, 500, 100])
    conn.close()

    resp = client.post(f"/groups/{gid}/keep")
    assert resp.status_code == 200
    assert "reviewed" in resp.text  # the card now shows reviewed

    conn = connect(config.db_path)
    actions = [r["action"] for r in conn.execute("SELECT action FROM decisions")]
    conn.close()
    assert actions == ["keep", "keep", "keep"]  # nothing trashed


def test_auto_resolve_keeps_largest_in_unreviewed(client, config):
    conn = connect(config.db_path)
    g1 = make_exact_group(conn, "a", [5000, 200])
    g2 = make_exact_group(conn, "b", [10, 9000])
    # g1 already manually reviewed (both kept) — auto must NOT touch it
    for pid in photo_ids(conn, g1):
        conn.execute(
            "INSERT INTO decisions (photo_id, action, decided_at) "
            "VALUES (?, 'keep', ?)",
            (pid, now()),
        )
    conn.commit()
    conn.close()

    resp = client.post("/review/auto", params={"tier": "exact"}, follow_redirects=False)
    assert resp.status_code == 303

    conn = connect(config.db_path)
    acts = {
        r["photo_id"]: r["action"]
        for r in conn.execute("SELECT photo_id, action FROM decisions")
    }
    g1_ids = photo_ids(conn, g1)
    g2_rows = conn.execute(
        """
        SELECT p.id, p.size FROM group_members m JOIN photos p ON p.id = m.photo_id
        WHERE m.group_id = ? ORDER BY p.size DESC
        """,
        (g2,),
    ).fetchall()
    conn.close()
    # g1 untouched: both still keep
    assert all(acts[pid] == "keep" for pid in g1_ids)
    # g2 auto-resolved: largest kept, the rest trashed
    assert acts[g2_rows[0]["id"]] == "keep"
    assert acts[g2_rows[1]["id"]] == "trash"


def test_reviewed_filter_and_space_reclaimable(client, config):
    conn = connect(config.db_path)
    gid = make_exact_group(conn, "a", [2000, 500])
    make_exact_group(conn, "b", [3000, 100])  # unreviewed
    big, small = photo_ids(conn, gid)
    conn.close()
    client.post(
        f"/groups/{gid}/decisions",
        data={f"action_{big}": "keep", f"action_{small}": "trash"},
    )

    page = client.get("/review", params={"tier": "exact"})
    assert "1 of 2 groups reviewed" in page.text
    assert "frees up" in page.text  # reclaimable space shown (500 bytes trashed)

    to_review = client.get("/review", params={"tier": "exact", "reviewed": "no"})
    assert "b-0.jpg" in to_review.text  # the undecided group
    assert "a-0.jpg" not in to_review.text  # the reviewed one is hidden
