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


def test_auto_resolve_never_clobbers_partial_manual_choices(client, config):
    """A group the user has started (even one decision) must be left alone —
    auto-resolve only touches fully-untouched groups."""
    conn = connect(config.db_path)
    gid = make_exact_group(conn, "a", [9000, 100, 50])  # big, small, tiny
    big, small, tiny = photo_ids(conn, gid)
    conn.close()
    # deliberately keep the SMALL one, trash the big; leave tiny undecided
    client.post(
        f"/groups/{gid}/decisions",
        data={f"action_{small}": "keep", f"action_{big}": "trash"},
    )

    client.post("/review/auto", params={"tier": "exact"}, follow_redirects=False)

    conn = connect(config.db_path)
    acts = {
        r["photo_id"]: r["action"]
        for r in conn.execute("SELECT photo_id, action FROM decisions")
    }
    conn.close()
    # the manual choices survive; the partially-decided group is untouched
    assert acts[small] == "keep"
    assert acts[big] == "trash"
    assert tiny not in acts  # still undecided, not force-trashed


def test_keep_group_survives_missing_group_row(client, config):
    """If the group is gone (e.g. a scan rebuilt it mid-request), respond
    gracefully instead of 500ing."""
    conn = connect(config.db_path)
    pid = insert_photo(conn, "x", md5="m")
    # a member row whose group row is gone — the transient state a rebuild
    # leaves between reading members and re-rendering (FK off to construct it)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        "INSERT INTO group_members (group_id, photo_id) VALUES (999, ?)", (pid,)
    )
    conn.commit()
    conn.close()

    resp = client.post("/groups/999/keep")
    assert resp.status_code == 200  # graceful placeholder, not a crash
    assert "changed during a scan" in resp.text


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


def test_group_confidence_helper():
    from doppel.app import group_confidence

    assert group_confidence("exact", [None, None]) == 1.0  # byte-identical
    # near: worst hamming of 8 over 64 bits -> ~0.875
    assert abs(group_confidence("near", [0, 8]) - (1 - 8 / 64)) < 1e-9
    # similar: worst (min) cosine
    assert group_confidence("similar", [0.99, 0.93]) == 0.93
    assert group_confidence("vlm", [None]) is None


def make_scored_group(conn, tier, key, scores):
    """A group in `tier` whose members carry the given scores vs anchor."""
    cur = conn.execute(
        "INSERT INTO groups (tier, created_at) VALUES (?, ?)", (tier, now())
    )
    gid = cur.lastrowid
    for i, score in enumerate(scores):
        pid = insert_photo(conn, f"{tier}-{key}-{i}", md5=f"{tier}{key}{i}", size=1000)
        conn.execute(
            "INSERT INTO group_members (group_id, photo_id, score) VALUES (?, ?, ?)",
            (gid, pid, score),
        )
    conn.commit()
    return gid


def test_similar_groups_sorted_by_confidence_desc(client, config):
    conn = connect(config.db_path)
    # a weaker group (min cosine 0.93) and a stronger one (min cosine 0.99)
    weak = make_scored_group(conn, "similar", "weak", [1.0, 0.93])
    strong = make_scored_group(conn, "similar", "strong", [1.0, 0.99])
    conn.close()

    page = client.get("/review", params={"tier": "similar"}).text
    assert "99% match" in page and "93% match" in page
    # the strongest match sorts to the top
    assert page.index(f"Group {strong} ·") < page.index(f"Group {weak} ·")


def test_near_groups_sorted_by_confidence_desc(client, config):
    conn = connect(config.db_path)
    loose = make_scored_group(conn, "near", "loose", [0, 8])  # worst hamming 8 -> 88%
    tight = make_scored_group(conn, "near", "tight", [0, 1])  # worst 1 -> 98%
    conn.close()

    page = client.get("/review", params={"tier": "near"}).text
    # tightest (highest confidence) first
    assert page.index(f"Group {tight} ·") < page.index(f"Group {loose} ·")


def test_review_mode_all_loads_everything(client, config):
    from doppel.app import REVIEW_BATCH

    conn = connect(config.db_path)
    for i in range(REVIEW_BATCH + 5):
        make_exact_group(conn, f"g{i}", [100, 10])
    conn.close()

    # default (scroll): only the first batch, plus a load-more sentinel
    scroll = client.get("/review", params={"tier": "exact"})
    assert "loading more" in scroll.text

    # switch to all-at-once
    client.post("/settings", data={"review_mode": "all"})
    everything = client.get("/review", params={"tier": "exact"})
    assert "loading more" not in everything.text  # no infinite-scroll sentinel
    # every group is on the page
    assert everything.text.count('class="review-group') == REVIEW_BATCH + 5


def test_sort_helpers():
    from doppel.app import default_sort, resolve_sort, sort_options

    assert default_sort("near") == "confidence"
    assert default_sort("exact") == "reclaim"
    # confidence is meaningless without a score, so it's coerced away off near/similar
    assert resolve_sort("exact", "confidence") == "reclaim"
    assert resolve_sort("near", "confidence") == "confidence"
    assert resolve_sort("exact", "garbage") == "reclaim"  # unknown -> tier default
    assert resolve_sort("similar", "reclaim") == "reclaim"
    assert "confidence" not in dict(sort_options("exact"))
    assert "confidence" in dict(sort_options("near"))


def test_review_sort_is_whitelisted_against_injection(client, config):
    conn = connect(config.db_path)
    make_exact_group(conn, "a", [2000, 500])
    conn.close()
    # a sort key crafted as SQL must be coerced to a safe default, not executed
    resp = client.get(
        "/review/pane",
        params={"tier": "exact", "sort": "reclaim); DROP TABLE photos;--"},
    )
    assert resp.status_code == 200
    conn = connect(config.db_path)
    # the photos table is untouched — no injection ran
    assert conn.execute("SELECT COUNT(*) AS n FROM photos").fetchone()["n"] == 2
    conn.close()


def test_review_sort_reclaim_orders_by_space_saved(client, config):
    conn = connect(config.db_path)
    small = make_exact_group(conn, "small", [1000, 900])  # frees ~900
    big = make_exact_group(conn, "big", [50000, 40000])  # frees ~40000
    conn.close()
    page = client.get("/review/pane", params={"tier": "exact", "sort": "reclaim"}).text
    assert page.index(f"Group {big} ·") < page.index(f"Group {small} ·")


def test_review_variants_filter(client, config):
    conn = connect(config.db_path)
    plain = make_scored_group(conn, "near", "plain", [0, 2])
    variant = make_scored_group(conn, "near", "var", [0, 3])
    conn.execute("UPDATE groups SET color_variant = 1 WHERE id = ?", (variant,))
    conn.commit()
    conn.close()

    all_near = client.get("/review/pane", params={"tier": "near"}).text
    assert f"Group {plain} ·" in all_near and f"Group {variant} ·" in all_near

    only_variants = client.get(
        "/review/pane", params={"tier": "near", "variants": "1"}
    ).text
    assert f"Group {variant} ·" in only_variants
    assert f"Group {plain} ·" not in only_variants


def test_review_pane_pushes_full_reloadable_url(client, config):
    conn = connect(config.db_path)
    make_scored_group(conn, "near", "g", [0, 3])
    conn.close()
    resp = client.get(
        "/review/pane",
        params={"tier": "near", "reviewed": "no", "sort": "reclaim", "variants": "1"},
    )
    assert resp.status_code == 200
    # htmx pushes this to the address bar so refresh/bookmark keep the view
    push = resp.headers.get("HX-Push-Url")
    assert push and push.startswith("/review?")
    for part in ("tier=near", "reviewed=no", "sort=reclaim", "variants=1"):
        assert part in push


def test_scroll_sentinel_carries_normalized_variants(client, config):
    conn = connect(config.db_path)
    for i in range(REVIEW_BATCH + 2):
        make_scored_group(conn, "near", f"g{i}", [0, 1])
    conn.close()
    # variants=0 is falsy: the filter is OFF, so the load-more sentinel must
    # not flip to variants=1 on the next page (page-1/page-N must agree)
    page = client.get("/review/pane", params={"tier": "near", "variants": "0"}).text
    assert "loading more" in page  # a sentinel exists
    assert "variants=1" not in page


def test_review_pane_returns_fragment(client, config):
    conn = connect(config.db_path)
    make_exact_group(conn, "a", [2000, 500])
    conn.close()
    resp = client.get("/review/pane", params={"tier": "exact"})
    assert resp.status_code == 200
    assert "review-controls" in resp.text  # the filter/sort bar
    assert "review-group" in resp.text  # the group cards
    assert "<html" not in resp.text.lower()  # a fragment, not a whole page


def test_workspace_landing_has_no_review_until_a_tier_is_picked(client, config):
    conn = connect(config.db_path)
    make_exact_group(conn, "a", [2000, 500])
    conn.close()
    # bare "/" is the dashboard: scan status, no split, no group cards yet
    home = client.get("/").text
    assert 'id="group-' not in home  # no rendered group cards
    assert "tier-nav" in home  # the category navigation is present
    assert "workspace split" not in home  # not split until a tier is chosen
    # picking a tier via the query opens the split with the review rendered
    split = client.get("/", params={"tier": "exact"}).text
    assert 'id="group-' in split
    assert "workspace split" in split


def test_review_card_shows_folder_path(client, config):
    conn = connect(config.db_path)
    cur = conn.execute(
        "INSERT INTO groups (tier, created_at) VALUES ('exact', ?)", (now(),)
    )
    gid = cur.lastrowid
    for i in range(2):
        pid = insert_photo(
            conn,
            f"fp-{i}",
            name=f"fp-{i}.jpg",
            md5="fp",
            size=1000 - i,
            folder_path="Photos / 2024 / Beach",
        )
        conn.execute(
            "INSERT INTO group_members (group_id, photo_id) VALUES (?, ?)", (gid, pid)
        )
    conn.commit()
    conn.close()

    page = client.get("/review", params={"tier": "exact"}).text
    assert "Photos / 2024 / Beach" in page


def test_settings_toggle_persists(client, config):
    from doppel.db import connect as db_connect
    from doppel.db import get_meta

    resp = client.post("/settings", data={"review_mode": "all"}, follow_redirects=False)
    assert resp.status_code == 303
    conn = db_connect(config.db_path)
    assert get_meta(conn, "review_mode") == "all"
    conn.close()
    assert "All at once" in client.get("/settings").text
