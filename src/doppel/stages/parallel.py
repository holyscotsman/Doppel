"""Shared concurrency primitives for the detection stages.

The one safety rule: **exactly one thread ever writes to a stage's sqlite
connection.** Worker threads only do the parallelizable work — fetch a
thumbnail, hash it, embed it — and hand finished ``(sql, params)`` tuples to a
single :class:`DbWriter` thread. That writer is the sole caller of
``conn.execute`` for those writes and commits in batches. This keeps stages
parallel (saturating CPU/GPU/network) while staying write-safe under WAL.

The stage's own setup/teardown writes (start_scan, the final grouping, and
finish_scan) run on the main thread *before the writer starts* and *after it is
joined* — never concurrently with it.
"""

from __future__ import annotations

import queue
import sqlite3
import threading
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


def parallel_map[T, R](
    fn: Callable[[T], R],
    items: Iterable[T],
    workers: int,
) -> Iterator[tuple[T, R | Exception]]:
    """Run ``fn(item)`` across a thread pool, yielding ``(item, result)`` in
    completion order. A single item's failure is never fatal: its exception is
    yielded as the value so the caller can skip it (and let a later run retry).

    workers <= 1 runs inline (deterministic, no threads) — handy for tests and
    for proving stage output is identical serial vs parallel.
    """
    items = list(items)
    if not items:
        return
    workers = max(1, workers)
    if workers == 1:
        for item in items:
            try:
                yield item, fn(item)
            except Exception as exc:  # noqa: BLE001 — surfaced as a value, per contract
                yield item, exc
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                yield item, future.result()
            except Exception as exc:  # noqa: BLE001
                yield item, exc


class DbWriter:
    """The single writer thread for a stage.

    Worker/main threads call :meth:`put` with pre-built ``(sql, params)``
    tuples; a background thread drains them and executes against ``conn`` in
    transactions of ``db_batch`` rows, committing per batch. Use as a context
    manager — on exit it flushes the tail, joins, and re-raises any error the
    writer hit so the stage's failure handling records it:

        with DbWriter(conn, db_batch, queue_maxsize) as writer:
            for item, result in parallel_map(work, items, workers):
                writer.put(sql, params)

    ``put`` never deadlocks if the writer dies: it detects the dead thread and
    re-raises instead of blocking forever on a full queue.
    """

    _SENTINEL = object()

    def __init__(
        self, conn: sqlite3.Connection, db_batch: int, queue_maxsize: int
    ) -> None:
        self._conn = conn
        self._batch = max(1, db_batch)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, queue_maxsize))
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> DbWriter:
        self._thread.start()
        return self

    def put(self, sql: str, params: tuple[Any, ...]) -> None:
        """Enqueue one write. Raises if the writer thread has already died so a
        full queue can never wedge a producer."""
        while True:
            if self._error is not None:
                raise self._error
            try:
                self._queue.put((sql, params), timeout=0.5)
                return
            except queue.Full:
                if not self._thread.is_alive():
                    raise self._error or RuntimeError("db writer thread died") from None

    def _run(self) -> None:
        pending: list[tuple[str, tuple[Any, ...]]] = []
        try:
            while True:
                item = self._queue.get()
                if item is self._SENTINEL:
                    break
                pending.append(item)
                if len(pending) >= self._batch:
                    self._flush(pending)
                    pending = []
            self._flush(pending)  # tail
        except BaseException as exc:  # noqa: BLE001 — reported to the main thread
            self._error = exc

    def _flush(self, pending: list[tuple[str, tuple[Any, ...]]]) -> None:
        if not pending:
            return
        for sql, params in pending:
            self._conn.execute(sql, params)
        self._conn.commit()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            self._queue.put(self._SENTINEL, timeout=5)
        except queue.Full:
            pass
        self._thread.join(timeout=60)
        # surface a writer-side failure, but don't mask an error already
        # propagating out of the with-block
        if self._error is not None and exc_type is None:
            raise self._error
