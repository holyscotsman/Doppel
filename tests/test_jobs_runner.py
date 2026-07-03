"""JobRunner's one-job-at-a-time contract."""

import threading

from doppel.jobs import JobRunner


def test_second_start_rejected_while_running() -> None:
    runner = JobRunner()
    release = threading.Event()
    started = threading.Event()

    def job() -> None:
        started.set()
        release.wait(timeout=10)

    assert runner.start("near", job) is True
    started.wait(timeout=10)
    assert runner.running_stage() == "near"
    assert runner.start("exact", lambda: None) is False

    release.set()
    runner.wait(timeout=10)
    assert runner.running_stage() is None
    assert runner.start("exact", lambda: None) is True
    runner.wait(timeout=10)


def test_job_exception_does_not_poison_runner() -> None:
    runner = JobRunner()

    def boom() -> None:
        raise RuntimeError("job blew up")

    assert runner.start("near", boom) is True
    runner.wait(timeout=10)
    assert runner.running_stage() is None
    assert runner.start("near", lambda: None) is True
    runner.wait(timeout=10)
