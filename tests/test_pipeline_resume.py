"""Phase-0 reliability: the 'all' pipeline resumes at the failed stage instead
of replaying Stage 1 (the slow Drive re-list), and the ETA uses a sliding
window rather than a startup-skewed cumulative average."""

from __future__ import annotations

from doppel.app import PIPELINE_STAGES, _pipeline_start_index, _RateEstimator
from doppel.db import connect


def _record_scan(conn, stage: str, status: str) -> None:
    conn.execute(
        "INSERT INTO scans (stage, status, processed, started_at) "
        "VALUES (?, ?, 0, 't')",
        (stage, status),
    )
    conn.commit()


def test_fresh_run_starts_at_sync(tmp_path) -> None:
    conn = connect(tmp_path / "d.db")
    assert _pipeline_start_index(conn) == 0  # nothing done yet


def test_resume_skips_completed_leading_stages(tmp_path) -> None:
    conn = connect(tmp_path / "d.db")
    _record_scan(conn, "sync", "done")
    _record_scan(conn, "exact", "done")
    _record_scan(conn, "near", "failed")  # crashed here
    idx = _pipeline_start_index(conn)
    assert idx == 2 and PIPELINE_STAGES[idx] == "near"  # resume at near, skip sync


def test_resume_at_similar_after_near_done(tmp_path) -> None:
    conn = connect(tmp_path / "d.db")
    for st in ("sync", "exact", "near"):
        _record_scan(conn, st, "done")
    _record_scan(conn, "similar", "failed")
    idx = _pipeline_start_index(conn)
    assert idx == 3 and PIPELINE_STAGES[idx] == "similar"


def test_fully_completed_pipeline_restarts_at_sync(tmp_path) -> None:
    conn = connect(tmp_path / "d.db")
    for st in PIPELINE_STAGES:
        _record_scan(conn, st, "done")
    # a manual 'run all' after a complete run re-syncs to catch new Drive files
    assert _pipeline_start_index(conn) == 0


def test_unfinished_sync_forces_full_restart(tmp_path) -> None:
    conn = connect(tmp_path / "d.db")
    _record_scan(conn, "sync", "failed")
    _record_scan(conn, "near", "failed")
    assert _pipeline_start_index(conn) == 0  # can't resume on a bad inventory


def test_resume_walks_an_effective_pipeline_with_an_appended_stage(tmp_path) -> None:
    # Brandarr appends 'classify' to the 'all' run when brand folders are set; a
    # classify-only failure must resume at classify, not replay the sync re-list
    conn = connect(tmp_path / "d.db")
    for st in PIPELINE_STAGES:
        _record_scan(conn, st, "done")
    _record_scan(conn, "classify", "failed")
    effective = [*PIPELINE_STAGES, "classify"]
    idx = _pipeline_start_index(conn, effective)
    assert effective[idx] == "classify"


def test_rate_estimator_uses_recent_window() -> None:
    est = _RateEstimator()
    assert est.rate("near", 1, 0, 0.0) is None  # one sample -> no rate yet
    assert est.rate("near", 1, 10, 10.0) == 1.0  # 10 items over 10s = 1/s


def test_rate_estimator_resets_on_new_scan() -> None:
    est = _RateEstimator()
    est.rate("near", 1, 0, 0.0)
    est.rate("near", 1, 100, 10.0)
    # a new scan of the same stage must not inherit the old scan's samples
    assert est.rate("near", 2, 0, 0.0) is None


def test_rate_estimator_drops_stale_samples() -> None:
    est = _RateEstimator()
    est.rate("near", 1, 0, 0.0)  # far outside the window
    est.rate("near", 1, 5, 100.0)  # slow early sample, now stale
    est.rate("near", 1, 105, 110.0)  # 100 items over the 10s since the last
    # only the last two samples are within WINDOW_S of each other -> ~10/s,
    # NOT the startup-skewed cumulative 105/110 ≈ 0.95/s
    assert est.rate("near", 1, 205, 120.0) == 10.0
