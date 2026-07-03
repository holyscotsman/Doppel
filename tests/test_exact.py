from doppel.stages.exact import run_exact
from tests.fakes import insert_photo


def groups_with_members(conn) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for row in conn.execute(
        """
        SELECT g.id, p.drive_id FROM groups g
        JOIN group_members m ON m.group_id = g.id
        JOIN photos p ON p.id = m.photo_id
        WHERE g.tier = 'exact'
        """
    ):
        out.setdefault(row["id"], set()).add(row["drive_id"])
    return out


def test_exact_groups_by_md5(conn) -> None:
    insert_photo(conn, "a1", md5="aaa")
    insert_photo(conn, "a2", md5="aaa")
    insert_photo(conn, "b1", md5="bbb")
    insert_photo(conn, "b2", md5="bbb")
    insert_photo(conn, "b3", md5="bbb")
    insert_photo(conn, "solo", md5="ccc")

    run_exact(conn)

    member_sets = sorted(groups_with_members(conn).values(), key=len)
    assert member_sets == [{"a1", "a2"}, {"b1", "b2", "b3"}]


def test_exact_ignores_null_md5_and_missing_photos(conn) -> None:
    insert_photo(conn, "n1", md5=None)
    insert_photo(conn, "n2", md5=None)
    insert_photo(conn, "m1", md5="mmm")
    insert_photo(conn, "m2", md5="mmm", status="missing")

    run_exact(conn)

    assert groups_with_members(conn) == {}


def test_exact_rebuild_is_idempotent(conn) -> None:
    insert_photo(conn, "a1", md5="aaa")
    insert_photo(conn, "a2", md5="aaa")

    run_exact(conn)
    run_exact(conn)

    assert len(groups_with_members(conn)) == 1
    n_members = conn.execute("SELECT COUNT(*) AS n FROM group_members").fetchone()["n"]
    assert n_members == 2


def test_exact_records_scan_progress(conn) -> None:
    insert_photo(conn, "a1", md5="aaa")
    insert_photo(conn, "a2", md5="aaa")

    scan_id = run_exact(conn)

    scan = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
    assert scan["stage"] == "exact"
    assert scan["status"] == "done"
    assert scan["processed"] == 1
    assert scan["total"] == 1


def test_decisions_survive_rebuild(conn) -> None:
    pid = insert_photo(conn, "a1", md5="aaa")
    insert_photo(conn, "a2", md5="aaa")
    run_exact(conn)
    conn.execute(
        "INSERT INTO decisions (photo_id, action, decided_at) VALUES (?, 'trash', 'x')",
        (pid,),
    )
    conn.commit()

    run_exact(conn)

    row = conn.execute(
        "SELECT action FROM decisions WHERE photo_id = ?", (pid,)
    ).fetchone()
    assert row["action"] == "trash"
