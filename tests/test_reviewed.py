"""Phase 1 (#1/#2): the explicit Reviewed toggle and bulk review-all /
unreview-all, plus Move-to-Trash scoped to reviewed groups. 'Reviewed' rides on
the decisions table (group ids are rebuilt every scan)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from doppel.app import create_app
from doppel.db import connect
from tests.fakes import FakeImageFetcher, insert_photo


@pytest.fixture
def client(config):
    app = create_app(
        config=config, fetcher_factory=lambda cfg: FakeImageFetcher(cfg.cache_dir)
    )
    with TestClient(app) as c:
        c.app = app
        yield c


@pytest.fixture
def group(client, config):
    """An exact group of two photos; returns (group_id, big_id, small_id)."""
    conn = connect(config.db_path)
    big = insert_photo(conn, "big", name="big.jpg", md5="dup", size=5000)
    small = insert_photo(conn, "small", name="small.jpg", md5="dup", size=100)
    conn.close()
    client.post("/scans/exact")
    client.app.state.runner.wait(timeout=10)
    conn = connect(config.db_path)
    gid = conn.execute("SELECT id FROM groups").fetchone()["id"]
    conn.close()
    return gid, big, small


def _decisions(config) -> dict[int, str]:
    conn = connect(config.db_path)
    rows = {
        r["photo_id"]: r["action"]
        for r in conn.execute("SELECT photo_id, action FROM decisions")
    }
    conn.close()
    return rows


def test_checking_reviewed_commits_shown_selection(client, config, group) -> None:
    gid, big, small = group
    assert _decisions(config) == {}  # nothing decided yet

    resp = client.post(f"/groups/{gid}/reviewed?value=1")

    assert resp.status_code == 200
    assert _decisions(config) == {big: "keep", small: "trash"}  # shown default
    assert "/reviewed?value=0" in resp.text  # card now renders as reviewed


def test_unchecking_reviewed_clears_decisions(client, config, group) -> None:
    gid, big, small = group
    client.post(f"/groups/{gid}/reviewed?value=1")

    resp = client.post(f"/groups/{gid}/reviewed?value=0")

    assert resp.status_code == 200
    assert _decisions(config) == {}
    assert "/reviewed?value=1" in resp.text  # card back to unreviewed


def test_review_all_marks_every_group_in_tier(client, config, group) -> None:
    gid, big, small = group
    resp = client.post(
        "/review/reviewed-all?tier=exact", headers={"HX-Request": "true"}
    )
    assert resp.status_code == 200
    assert _decisions(config) == {big: "keep", small: "trash"}


def test_review_all_never_overwrites_a_manual_choice(client, config, group) -> None:
    gid, big, small = group
    # manually keep BOTH (not dupes); review-all must not flip this to trash
    client.post(
        f"/groups/{gid}/decisions",
        data={f"action_{big}": "keep", f"action_{small}": "keep"},
    )
    client.post("/review/reviewed-all?tier=exact", headers={"HX-Request": "true"})
    assert _decisions(config) == {big: "keep", small: "keep"}


def test_unreview_all_clears_the_tier(client, config, group) -> None:
    gid, big, small = group
    client.post("/review/reviewed-all?tier=exact", headers={"HX-Request": "true"})
    resp = client.post(
        "/review/unreviewed-all?tier=exact", headers={"HX-Request": "true"}
    )
    assert resp.status_code == 200
    assert _decisions(config) == {}


def test_move_to_trash_excludes_unreviewed_groups(client, config, group) -> None:
    gid, big, small = group
    # decide only the small one -> group is NOT fully reviewed (big undecided)
    client.post(f"/groups/{gid}/decisions", data={f"action_{small}": "trash"})
    page = client.get("/trash/confirm")
    assert "small.jpg" not in page.text  # its group isn't reviewed yet

    # review the whole group -> its trash-marked member becomes eligible
    client.post(f"/groups/{gid}/reviewed?value=1")
    page = client.get("/trash/confirm")
    assert "small.jpg" in page.text


# ---- #5 display modes ---------------------------------------------------


def test_scroll_is_the_default_mode(client, config, group) -> None:
    pane = client.get("/review/pane?tier=exact")
    assert "Continuous scroll" in pane.text  # the mode selector is present
    assert "Review this batch" not in pane.text  # batch controls are hidden


def test_batch_mode_makes_review_this_batch_the_primary_action(
    client, config, group
) -> None:
    resp = client.post(
        "/review/mode?mode=batch&tier=exact", headers={"HX-Request": "1"}
    )
    assert resp.status_code == 200
    # in batch mode the prominent action reviews only this batch, with an
    # explicit "all groups" option beside it (not a bare "Review all")
    assert "Review all in this batch" in resp.text
    assert "Review all 1 groups" in resp.text
    assert "/review/reviewed-batch" in resp.text


def test_review_this_batch_marks_the_page(client, config, group) -> None:
    gid, big, small = group
    client.post("/review/mode?mode=batch&tier=exact", headers={"HX-Request": "1"})
    resp = client.post(
        "/review/reviewed-batch?tier=exact&page=1", headers={"HX-Request": "1"}
    )
    assert resp.status_code == 200
    assert _decisions(config) == {big: "keep", small: "trash"}


def test_all_at_once_mode_has_no_batch_controls(client, config, group) -> None:
    client.post("/review/mode?mode=all&tier=exact", headers={"HX-Request": "1"})
    pane = client.get("/review/pane?tier=exact")
    assert "Review this batch" not in pane.text


def test_review_all_honors_the_color_variants_filter(client, config) -> None:
    """Review-all in a 'color variants only' view must NOT commit trash decisions
    on the non-variant groups it hides from the user."""
    conn = connect(config.db_path)
    ids = {}
    for k in ("va", "vb", "pa", "pb"):
        ids[k] = insert_photo(conn, k, name=f"{k}.jpg", size=100 if k[1] == "a" else 50)
    gv = conn.execute(
        "INSERT INTO groups (tier, color_variant, created_at) VALUES ('near', 1, 't')"
    ).lastrowid
    gp = conn.execute(
        "INSERT INTO groups (tier, color_variant, created_at) VALUES ('near', 0, 't')"
    ).lastrowid
    for g, a, b in ((gv, "va", "vb"), (gp, "pa", "pb")):
        conn.execute(
            "INSERT INTO group_members (group_id, photo_id, score) "
            "VALUES (?, ?, 0), (?, ?, 4)",
            (g, ids[a], g, ids[b]),
        )
    conn.commit()
    conn.close()

    client.post(
        "/review/reviewed-all?tier=near&variants=1", headers={"HX-Request": "1"}
    )

    decided = _decisions(config)
    assert ids["va"] in decided and ids["vb"] in decided  # variant group finalized
    assert (
        ids["pa"] not in decided and ids["pb"] not in decided
    )  # hidden group untouched
