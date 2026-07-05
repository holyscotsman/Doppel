"""Runtime diagnostics: rotating file logs + a native-crash dumper.

A segfault inside a C extension (torch/MPS, Pillow, sqlite, OpenSSL) kills the
process with signal 11 — `make run` reports exit 139 — and leaves NO Python
traceback, because the interpreter is already gone. `faulthandler` installs a
signal handler that writes the native stack of every thread to `logs/crash.log`
the instant that happens, so a crash can be read after the fact instead of
guessed at. The rotating app log records stage transitions, fetch failures, and
the `/thumb` errors so the run leading up to a crash is on disk too.

Called from `build()` (real server runs only); tests use `create_app()`
directly and never touch the filesystem here.
"""

from __future__ import annotations

import faulthandler
import logging
import logging.handlers
from pathlib import Path
from typing import IO

# The crash file must stay open for the whole process — faulthandler writes to
# its file descriptor on a fatal signal, so a closed/GC'd handle would lose the
# dump. Held at module scope for exactly that lifetime.
_crash_file: IO[str] | None = None


def setup_diagnostics(logs_dir: Path | str = "logs", app_name: str = "doppel") -> Path:
    """Enable the native-crash dumper and attach a rotating file log. Idempotent:
    safe to call more than once (handlers are de-duplicated). Returns the logs
    directory."""
    global _crash_file
    logs = Path(logs_dir)
    logs.mkdir(parents=True, exist_ok=True)

    if _crash_file is None:
        # append so successive crashes accumulate; line-buffered to flush early
        _crash_file = open(logs / "crash.log", "a", buffering=1)  # noqa: SIM115
        faulthandler.enable(file=_crash_file, all_threads=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(getattr(h, "_doppel_file", False) for h in root.handlers):
        handler = logging.handlers.RotatingFileHandler(
            logs / f"{app_name}.log",
            maxBytes=5_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"
            )
        )
        handler._doppel_file = True  # type: ignore[attr-defined]
        root.addHandler(handler)

    logging.getLogger(app_name).info(
        "diagnostics ready — app log + crash dumps in %s/", logs.resolve()
    )
    return logs
