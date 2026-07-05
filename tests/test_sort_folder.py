"""Phase 1 (#3): the review preselect prefers trashing the copy in a *sort*
folder (substring, case-insensitive) and keeps the copy in a real folder;
ties fall back to keeping the largest. Members are always passed largest-first."""

from __future__ import annotations

from doppel.app import default_selection


def _m(pid: int, folder: str | None) -> dict:
    return {"id": pid, "folder_path": folder}


def test_keeps_real_folder_trashes_sort_inbox() -> None:
    members = [_m(1, "Photos / 2024 / Italy"), _m(2, "Photos / To Sort")]
    assert default_selection(members, True, "sort") == {1: "keep", 2: "trash"}


def test_prefers_non_sort_even_when_it_is_smaller() -> None:
    # member 1 (largest) is in a sort folder; keep the smaller real-folder copy
    members = [_m(1, "To Sort"), _m(2, "Camera Roll")]
    assert default_selection(members, True, "sort") == {1: "trash", 2: "keep"}


def test_substring_is_case_insensitive() -> None:
    # "Unsorted" contains "sort" — substring matching, as chosen
    members = [_m(1, "Unsorted"), _m(2, "Albums / Trip")]
    assert default_selection(members, True, "sort") == {1: "trash", 2: "keep"}


def test_all_in_sort_keeps_largest() -> None:
    members = [_m(1, "To Sort"), _m(2, "unsorted")]  # 1 is largest
    assert default_selection(members, True, "sort") == {1: "keep", 2: "trash"}


def test_none_in_sort_keeps_largest() -> None:
    members = [_m(1, "Albums"), _m(2, "Camera Roll")]
    assert default_selection(members, True, "sort") == {1: "keep", 2: "trash"}


def test_toggle_off_ignores_sort_and_keeps_largest() -> None:
    members = [_m(1, "To Sort"), _m(2, "Camera Roll")]
    assert default_selection(members, False, "sort") == {1: "keep", 2: "trash"}


def test_missing_folder_path_is_not_a_sort_folder() -> None:
    members = [_m(1, None), _m(2, "To Sort")]
    assert default_selection(members, True, "sort") == {1: "keep", 2: "trash"}


def test_empty_members() -> None:
    assert default_selection([], True, "sort") == {}
