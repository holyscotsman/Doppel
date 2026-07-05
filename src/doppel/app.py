"""FastAPI app: routes, templates, job wiring. Server-rendered UI (Jinja2 +
htmx); binds to 127.0.0.1 only (see Makefile run target)."""

from __future__ import annotations

import collections
import html
import logging
import sqlite3
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from doppel.config import Config, load_config, set_config_value
from doppel.db import connect, get_meta, set_meta
from doppel.drive import (
    DRIVE_WRITE_SCOPES,
    SERVICE_ACCOUNT_PATH,
    CredentialsRequired,
    FetchError,
    GoogleDriveClient,
    ImageFetcher,
    TrashNotAuthorized,
    get_credentials,
    is_service_account_key,
    load_service_account_credentials,
    parse_folder_input,
    service_account_email,
    web_auth_flow,
)
from doppel.embed import ClipEmbedder, Embedder
from doppel.jobs import (
    JobRunner,
    fail_scan,
    now,
    reconcile_orphaned_scans,
    run_sync,
    start_scan,
)
from doppel.stages.adjudicate import run_adjudicate
from doppel.stages.exact import run_exact
from doppel.stages.near import run_near
from doppel.stages.similar import run_similar
from doppel.vlm import OllamaClient, VlmClient

log = logging.getLogger("doppel")

PACKAGE_DIR = Path(__file__).parent

# stages the UI can launch, in pipeline order; extended phase by phase
UI_STAGES = ["sync", "exact", "near", "similar", "adjudicate"]

# the core detection pipeline the "Run full scan" button chains, in order
PIPELINE_STAGES = ["sync", "exact", "near", "similar"]


def _pipeline_start_index(conn: sqlite3.Connection) -> int:
    """Where to begin the 'all' pipeline. A fresh run — or one whose last attempt
    fully completed — starts at sync (a full re-scan picks up new Drive files).
    A RESUME after a downstream failure skips the leading stages that are already
    'done' (above all the slow Drive re-list) and restarts at the first stage
    that isn't done, so a Stage-3 failure resumes at Stage 3 instead of replaying
    Stage 1."""
    status: dict[str, str | None] = {}
    for st in PIPELINE_STAGES:
        row = conn.execute(
            "SELECT status FROM scans WHERE stage = ? ORDER BY id DESC LIMIT 1",
            (st,),
        ).fetchone()
        status[st] = row["status"] if row else None
    # no completed sync yet, or the whole pipeline already finished -> full restart
    if status["sync"] != "done" or all(status[st] == "done" for st in PIPELINE_STAGES):
        return 0
    for i, st in enumerate(PIPELINE_STAGES):
        if status[st] != "done":
            return i
    return 0


# plain-language names for the dashboard (the stage keys are jargon)
STAGE_LABELS = {
    "sync": "Find photos in Drive",
    "exact": "Exact duplicates",
    "near": "Near-duplicates",
    "similar": "Similar photos",
    "adjudicate": "AI double-check",
}

PAGE_SIZE = 20

# groups loaded per infinite-scroll batch on the one-page review
REVIEW_BATCH = 8

REVIEWED_FILTERS = {
    "all": "",
    "yes": "HAVING decided = members",
    "no": "HAVING decided < members",
}


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _scan_timing(scan: dict | None) -> tuple[str | None, str | None]:
    """(elapsed, eta) as human strings for a scans row. ETA is a rate
    projection from processed/total and only shown while running."""
    if not scan or not scan.get("started_at"):
        return None, None
    try:
        started = datetime.fromisoformat(scan["started_at"])
    except ValueError:
        return None, None
    if scan.get("finished_at"):
        end = datetime.fromisoformat(scan["finished_at"])
    else:
        end = datetime.now(UTC)
    elapsed = (end - started).total_seconds()
    eta = None
    if (
        scan.get("status") == "running"
        and scan.get("total")
        and scan.get("processed")
        and elapsed > 0
    ):
        rate = scan["processed"] / elapsed  # items per second
        remaining = (scan["total"] - scan["processed"]) / rate if rate > 0 else 0
        if remaining > 0:
            eta = _fmt_duration(remaining)
    return _fmt_duration(max(elapsed, 0)), eta


class _RateEstimator:
    """Sliding-window throughput for a live ETA. The scans row only stores a
    running (processed, total); a cumulative processed/elapsed rate badly
    overestimates the ETA early, because the first samples are dominated by
    one-off startup cost (model load, first connections). This samples processed
    over wall-clock time and projects from the most recent WINDOW_S seconds, so
    the ETA tracks the true steady-state rate. Per-stage; resets when a new scan
    of that stage starts. Thread-safe (the UI polls it from request threads)."""

    WINDOW_S = 30.0

    def __init__(self) -> None:
        self._data: dict[str, dict] = {}
        self._lock = threading.Lock()

    def rate(
        self, stage: str, scan_id: int, processed: int, now_ts: float
    ) -> float | None:
        """Items/second over the recent window, or None until there are enough
        samples spanning real progress."""
        with self._lock:
            d = self._data.get(stage)
            if d is None or d["scan_id"] != scan_id:
                d = {"scan_id": scan_id, "samples": collections.deque(maxlen=600)}
                self._data[stage] = d
            samples: collections.deque = d["samples"]
            if not samples or samples[-1][1] != processed:
                samples.append((now_ts, processed))
            while len(samples) >= 2 and now_ts - samples[0][0] > self.WINDOW_S:
                samples.popleft()
            if len(samples) < 2:
                return None
            span = samples[-1][0] - samples[0][0]
            done = samples[-1][1] - samples[0][1]
            if span <= 0 or done <= 0:
                return None
            return done / span


HASH_BITS = 64  # pHash/dHash are 8x8 = 64-bit


def default_selection(members: list, prefer_sort: bool, keyword: str) -> dict[int, str]:
    """The pre-checked keep/trash choice for a group's members (passed
    largest-first). Keep exactly one photo, trash the rest. When prefer_sort is
    on, the kept photo is the largest one NOT in a folder whose path contains
    `keyword` (case-insensitive) — so a copy sitting in a "To Sort" inbox is
    trashed in favour of the copy filed in a real folder. If every copy (or no
    copy) is in such a folder, fall back to keeping the largest. Saved decisions
    always override this default."""
    if not members:
        return {}
    kw = keyword.lower().strip()

    def in_sort(m: object) -> bool:
        return bool(prefer_sort and kw and kw in (m["folder_path"] or "").lower())

    non_sort = [m for m in members if not in_sort(m)]
    keeper = (non_sort or members)[0]["id"]  # members are largest-first
    return {m["id"]: ("keep" if m["id"] == keeper else "trash") for m in members}


def group_confidence(tier: str, scores: list) -> float | None:
    """A 0-1 confidence that a group's members really are duplicates.

    exact = certain (byte-identical). near = from the WORST hash distance to
    the anchor (0 = identical, up to 64). similar = the WORST cosine to the
    anchor (already 0-1). vlm/unknown -> None (no numeric score)."""
    if tier == "exact":
        return 1.0
    numeric = [s for s in scores if s is not None]
    if not numeric:
        return None
    if tier == "near":
        return max(0.0, 1.0 - max(numeric) / HASH_BITS)
    if tier == "similar":
        return min(numeric)
    return None


# how to order each tier's groups in the review: most-confident first.
# confidence is set by the WORST member vs the anchor — for near that's the
# largest hash distance (max_score), for similar the lowest cosine (min_score).
REVIEW_ORDER = {
    "near": "ORDER BY max_score ASC, g.id",  # smallest worst-distance = tightest
    "similar": "ORDER BY min_score DESC, g.id",  # highest worst-cosine = tightest
}

# user-selectable sort options for the review pane. Each maps to a fixed
# ORDER BY fragment (never interpolated from user input) — the key is
# validated against this whitelist. "reclaim" = potential space freed by
# keeping only the largest member (total size minus the biggest file).
SORT_LABELS = {
    "confidence": "best match first",
    "reclaim": "biggest space savings",
    "largest": "most photos",
    "smallest": "fewest photos",
    "size": "largest files first",
}
_SIZE = "COALESCE(SUM(p2.size), 0)"
_RECLAIM = f"({_SIZE} - COALESCE(MAX(p2.size), 0))"
_SORT_ORDER = {
    "reclaim": f"ORDER BY {_RECLAIM} DESC, g.id",
    "largest": "ORDER BY members DESC, g.id",
    "smallest": "ORDER BY members ASC, g.id",
    "size": f"ORDER BY {_SIZE} DESC, g.id",
}


def _truthy(value: str) -> bool:
    return value.lower() in ("1", "true", "on", "yes")


def _pane_push_url(tier: str, reviewed: str, sort: str, variants: bool) -> str:
    """The full-page URL that reproduces a review-pane view, for the browser
    address bar. Refreshing/bookmarking it re-renders the same split workspace,
    so filter/sort/variants survive a reload."""
    from urllib.parse import urlencode

    params = {"tier": tier, "reviewed": reviewed, "sort": sort}
    if variants:
        params["variants"] = "1"
    return "/review?" + urlencode(params)


def default_sort(tier: str) -> str:
    """Confidence only means something where there's a numeric score."""
    return "confidence" if tier in ("near", "similar") else "reclaim"


def resolve_sort(tier: str, sort: str | None) -> str:
    """Coerce a sort key to something valid for this tier (cosmetic param —
    never 422; fall back to the tier default)."""
    if sort == "confidence" and tier in ("near", "similar"):
        return "confidence"
    if sort in _SORT_ORDER:
        return sort
    return default_sort(tier)


def sort_order_clause(tier: str, sort: str) -> str:
    if sort == "confidence":
        return REVIEW_ORDER.get(tier, "ORDER BY g.id")
    return _SORT_ORDER.get(sort, "ORDER BY g.id")


def sort_options(tier: str) -> list[tuple[str, str]]:
    """(key, label) sort choices offered for a tier — confidence only where a
    score exists."""
    keys = list(SORT_LABELS)
    if tier not in ("near", "similar"):
        keys = [k for k in keys if k != "confidence"]
    return [(k, SORT_LABELS[k]) for k in keys]


def scan_is_due(
    enabled: bool,
    last_run: datetime | None,
    now: datetime,
    period_hours: float = 24,
) -> bool:
    """Should the daily auto-scan fire? Only when enabled and the last scan
    was at least `period_hours` ago (or there was never one)."""
    if not enabled:
        return False
    if last_run is None:
        return True
    return (now - last_run).total_seconds() >= period_hours * 3600


class DailyScheduler:
    """A daemon thread that fires a scan roughly once a day while the app is
    running. It only nudges — the trigger no-ops if a job is already going or
    Drive isn't connected, and it re-checks every interval."""

    def __init__(
        self,
        is_due: Callable[[], bool],
        trigger: Callable[[], None],
        interval: float = 600.0,
    ) -> None:
        self._is_due = is_due
        self._trigger = trigger
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                if self._is_due():
                    self._trigger()
            except Exception:
                traceback.print_exc()


def auth_mode() -> str | None:
    """How Drive is connected: a service-account key takes precedence over an
    OAuth token (it's the cleaner path — no consent screen, no expiry)."""
    if Path(SERVICE_ACCOUNT_PATH).exists():
        return "service_account"
    if Path("token.json").exists():
        return "oauth"
    return None


def build_drive_credentials() -> object:
    """Credentials for the current auth mode. Raises CredentialsRequired when
    Drive isn't connected yet."""
    mode = auth_mode()
    if mode == "service_account":
        return load_service_account_credentials()
    if mode == "oauth":
        return get_credentials(allow_interactive=False)
    raise CredentialsRequired(
        "Google Drive is not connected yet — finish step 1 of the setup wizard."
    )


def build_trash_credentials() -> object:
    """Write-scoped credentials for the move-to-trash action ONLY. Everything
    else in the app uses the read-only credentials from build_drive_credentials.

    Service-account mode reads the same key with the write scope (it can still
    only touch folders shared with edit access). OAuth mode reuses the existing
    token but requires it to carry the write scope — a read-only connection
    raises TrashNotAuthorized so the UI can prompt a reconnect."""
    mode = auth_mode()
    if mode == "service_account":
        return load_service_account_credentials(scopes=DRIVE_WRITE_SCOPES)
    if mode == "oauth":
        # NOTE: the OAuth flow only ever requests the read-only SCOPES, and
        # get_credentials loads the token forcing those same scopes — so an
        # OAuth connection can never satisfy this check today (it fails closed
        # into TrashNotAuthorized, which is safe). Service-account mode is the
        # supported trash path. If OAuth trashing is ever wanted, add a
        # write-scoped consent flow AND load the token with its real granted
        # scopes, or this check will keep rejecting a genuinely-granted token.
        creds = get_credentials(allow_interactive=False)
        granted = set(getattr(creds, "scopes", None) or [])
        if not granted & set(DRIVE_WRITE_SCOPES):
            raise TrashNotAuthorized(
                "Google Drive is connected read-only. To move files to Trash, "
                "connect with a service account (recommended) and share your "
                "folder with edit access."
            )
        return creds
    raise CredentialsRequired(
        "Google Drive is not connected yet — finish step 1 of the setup wizard."
    )


def can_trash() -> bool:
    """Whether the current connection can move files to Trash — used to gate
    the confirm-page button. A service account is assumed able (real edit
    access is verified per-file at trash time); OAuth needs the write scope."""
    mode = auth_mode()
    if mode == "service_account":
        return True
    if mode == "oauth":
        try:
            creds = get_credentials(allow_interactive=False)
        except Exception:
            return False
        granted = set(getattr(creds, "scopes", None) or [])
        return bool(granted & set(DRIVE_WRITE_SCOPES))
    return False


def _resolve_scan_roots(client: object, config: Config) -> list[str] | None:
    """Which folders a sync should walk, given the auth mode and scope.

    - explicit folder scope -> just that folder
    - OAuth, no scope -> None (whole My Drive, one unscoped query)
    - service account, no scope -> every folder shared with it (its own My
      Drive is empty, so an unscoped query would find nothing and wrongly
      mark the whole inventory missing)
    """
    if config.drive_folder_id:
        return [config.drive_folder_id]
    if auth_mode() == "service_account":
        shared = client.list_shared_folders()
        if not shared:
            raise RuntimeError(
                "No folders are shared with the service account yet — in Google "
                "Drive, share a folder with the account's email, then sync."
            )
        return [f["id"] for f in shared]
    return None


def _build_real_fetcher(config: Config) -> ImageFetcher:
    from google.auth.transport.requests import AuthorizedSession

    from doppel.drive import DriveImageFetcher

    creds = build_drive_credentials()
    # a factory, not a single session: the fetcher builds one AuthorizedSession
    # per worker thread (requests sessions are not thread-safe).
    return DriveImageFetcher(
        db_path=config.db_path,
        client=GoogleDriveClient(creds),
        session_factory=lambda: AuthorizedSession(creds),
        cache_dir=config.cache_dir,
    )


# vision models that take one image per request and can't compare a pair, so
# they're useless for the adjudication stage — matched as substrings of the
# model name (before the tag). Every installed model reports the "vision"
# capability, so this denylist is what actually separates usable from not.
INCOMPATIBLE_VISION_MODELS = (
    "minicpm",
    "llava",
    "bakllava",
    "moondream",
    "llama3.2-vision",
    "llama-vision",
)


def _installed_model_names(client: object) -> list[str]:
    response = client.list()
    models = getattr(response, "models", None) or response.get("models", [])
    names = []
    for m in models:
        name = getattr(m, "model", None) or m.get("model") or m.get("name")
        if name:
            names.append(name)
    return names


def _usable_models(client: object) -> list[str]:
    """Filter an Ollama client's installed models to those usable for
    multi-image adjudication: vision-capable and not a known single-image
    family."""
    usable = []
    for name in _installed_model_names(client):
        base = name.split(":", 1)[0].lower()
        if any(bad in base for bad in INCOMPATIBLE_VISION_MODELS):
            continue
        try:
            caps = getattr(client.show(name), "capabilities", None) or []
        except Exception:
            continue  # model metadata unavailable — skip rather than guess
        if "vision" in caps:
            usable.append(name)
    return usable


def _list_ollama_models(host: str) -> list[str]:
    """Usable models on an Ollama server. Raises if the host is unreachable.

    A short timeout keeps /setup from hanging when the host is down (ollama's
    client defaults to no timeout); the client is closed so its connection
    pool doesn't leak across probes."""
    import ollama

    with ollama.Client(host=host, timeout=5) as client:
        return _usable_models(client)


def create_app(
    config: Config | None = None,
    fetcher_factory: Callable[[Config], ImageFetcher] | None = None,
    embedder_factory: Callable[[Config], Embedder] | None = None,
    vlm_factory: Callable[[Config], VlmClient] | None = None,
    config_path: Path | str = "config.toml",
    oauth_flow_factory: Callable[[str], object] | None = None,
    ollama_lister: Callable[[str], list[str]] | None = None,
    drive_client_factory: Callable[[], GoogleDriveClient] | None = None,
    trash_client_factory: Callable[[], GoogleDriveClient] | None = None,
    enable_scheduler: bool = False,
) -> FastAPI:
    """Build the app. Tests inject a Config and fake factories. The daily-scan
    scheduler thread only starts when enable_scheduler is set (real runs)."""
    config = config or load_config(config_path)
    fetcher_factory = fetcher_factory or _build_real_fetcher
    embedder_factory = embedder_factory or (lambda cfg: ClipEmbedder(cfg.clip_model))
    vlm_factory = vlm_factory or (
        lambda cfg: OllamaClient(cfg.ollama.host, cfg.ollama.model)
    )
    oauth_flow_factory = oauth_flow_factory or (
        lambda redirect_uri: web_auth_flow("credentials.json", redirect_uri)
    )
    ollama_lister = ollama_lister or _list_ollama_models
    drive_client_factory = drive_client_factory or (
        lambda: GoogleDriveClient(build_drive_credentials())
    )
    trash_client_factory = trash_client_factory or (
        lambda: GoogleDriveClient(build_trash_credentials())
    )

    app = FastAPI(title="doppel")
    app.state.config = config
    app.state.runner = JobRunner()
    app.state.rate = _RateEstimator()  # sliding-window ETA for running stages
    app.state.fetcher = None  # built lazily: needs OAuth credentials
    app.state.embedder = None  # built lazily: loads the CLIP model
    app.state.vlm = None  # built lazily: needs the Ollama server
    app.state.oauth = None  # in-flight wizard OAuth state

    def reload_config() -> None:
        """Re-read config.toml after the wizard writes it and drop lazy
        singletons built from the old settings. Holds init_lock so it cannot
        race a concurrent get_*() that would otherwise re-cache a client
        built from the pre-reload config."""
        nonlocal config
        with init_lock:
            config = load_config(config_path)
            app.state.config = config
            app.state.fetcher = None
            app.state.embedder = None
            app.state.vlm = None

    templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))
    app.mount(
        "/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static"
    )

    @app.middleware("http")
    async def same_origin_guard(request: Request, call_next):
        """CSRF defense for a session-less localhost app: reject mutating
        requests whose Origin/Referer is a different host than ours. A
        browser always attaches Origin to a cross-origin POST, so a
        malicious page cannot forge writes; programmatic clients (curl, the
        test suite) send neither header and are allowed through."""
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            source = request.headers.get("origin") or request.headers.get("referer")
            if source is not None:
                from urllib.parse import urlparse

                if urlparse(source).netloc != request.url.netloc:
                    return Response("cross-origin request refused", status_code=403)
        return await call_next(request)

    # a killed process leaves 'running' scans rows behind; at startup no job
    # can actually be running, so repair the ledger once
    startup_conn = connect(config.db_path)
    try:
        reconcile_orphaned_scans(startup_conn)
    finally:
        startup_conn.close()

    def get_conn() -> Iterator[sqlite3.Connection]:
        conn = connect(config.db_path)
        try:
            yield conn
        finally:
            conn.close()

    # request handlers run on threadpool threads: lazy singletons need a lock
    init_lock = threading.Lock()

    def get_fetcher() -> ImageFetcher:
        with init_lock:
            if app.state.fetcher is None:
                try:
                    app.state.fetcher = fetcher_factory(config)
                except CredentialsRequired as exc:
                    raise HTTPException(status_code=503, detail=str(exc)) from exc
            return app.state.fetcher

    def get_embedder() -> Embedder:
        with init_lock:
            if app.state.embedder is None:
                app.state.embedder = embedder_factory(config)
            return app.state.embedder

    def get_vlm() -> VlmClient:
        with init_lock:
            if app.state.vlm is None:
                app.state.vlm = vlm_factory(config)
            return app.state.vlm

    def scan_overview(conn: sqlite3.Connection) -> list[dict]:
        """Latest scans row per UI stage, with a plain-language label, elapsed
        time, and an ETA for a running stage. Which stage is live comes from
        its scans-row status (works for a single stage and the 'all' pipeline)."""
        overview = []
        for stage in UI_STAGES:
            row = conn.execute(
                "SELECT * FROM scans WHERE stage = ? ORDER BY id DESC LIMIT 1",
                (stage,),
            ).fetchone()
            scan = dict(row) if row else None
            elapsed, _ = _scan_timing(scan)  # elapsed only; ETA is windowed below
            state = scan["status"] if scan else "idle"
            # ETA from the sliding-window rate (see _RateEstimator). Show
            # "estimating…" while a stage runs but has no usable ETA yet: either
            # the window has too little data, or the total isn't known (the sync
            # stage paginates and only learns its total at the end).
            eta: str | None = None
            estimating = False
            if state == "running" and scan:
                if scan["total"] and scan["processed"]:
                    rate = app.state.rate.rate(
                        stage, scan["id"], scan["processed"], time.monotonic()
                    )
                    if rate:
                        remaining = (scan["total"] - scan["processed"]) / rate
                        if remaining > 0:
                            eta = _fmt_duration(remaining)
                estimating = eta is None
            # 0-1 fill for the progress ring; None while a running stage has no
            # known total yet (the ring shows an indeterminate spinner instead).
            if state == "done":
                fraction: float | None = 1.0
            elif state == "running" and scan and scan["total"]:
                fraction = min(1.0, scan["processed"] / scan["total"])
            elif state == "running":
                fraction = None
            else:
                fraction = 0.0
            overview.append(
                {
                    "stage": stage,
                    "label": STAGE_LABELS[stage],
                    "scan": scan,
                    "elapsed": elapsed,
                    "eta": eta,
                    "estimating": estimating,
                    "state": state,
                    "fraction": fraction,
                    "pct": None if fraction is None else round(fraction * 100),
                }
            )
        return overview

    def _build_stage_callable(
        conn: sqlite3.Connection, stage: str
    ) -> Callable[[], object]:
        """Acquire a stage's dependencies and return a zero-arg runner. Raises
        if a dependency (credentials, model, shared folders) is unavailable."""
        if stage == "sync":
            client = drive_client_factory()
            folder_ids = _resolve_scan_roots(client, config)
            return lambda: run_sync(conn, client, config.cache_dir, folder_ids)
        if stage == "exact":
            return lambda: run_exact(conn)
        if stage == "near":
            fetcher = get_fetcher()
            return lambda: run_near(conn, fetcher, config)
        if stage == "similar":
            fetcher, embedder = get_fetcher(), get_embedder()
            return lambda: run_similar(conn, fetcher, embedder, config)
        if stage == "adjudicate":
            fetcher, vlm = get_fetcher(), get_vlm()
            return lambda: run_adjudicate(conn, fetcher, vlm, config)
        raise ValueError(f"unknown stage {stage!r}")

    def run_stage_job(stage: str) -> None:
        """Run one stage, or the whole detection pipeline for stage == 'all'.
        The interactive OAuth flow is never run here: it would block the worker
        forever if the consent tab is missed."""
        conn = connect(config.db_path)
        try:
            if stage == "all":
                start = _pipeline_start_index(conn)
                stages = PIPELINE_STAGES[start:]
                if start > 0:
                    log.info(
                        "resuming pipeline at %r (skipping completed %s)",
                        stages[0],
                        PIPELINE_STAGES[:start],
                    )
            else:
                stages = [stage]
            for st in stages:
                # a dependency failure happens before the stage's own
                # start_scan, so record it in the ledger — otherwise the click
                # would look like a silent no-op in the UI
                try:
                    job = _build_stage_callable(conn, st)
                except Exception as exc:
                    scan_id = start_scan(conn, st)
                    detail = exc.detail if isinstance(exc, HTTPException) else exc
                    fail_scan(conn, scan_id, f"{type(exc).__name__}: {detail}")
                    log.warning("stage %s could not start: %s", st, detail)
                    return  # stop the pipeline on a dependency failure
                try:
                    log.info("stage %s starting", st)
                    job()
                    log.info("stage %s finished", st)
                except Exception:
                    # the stage recorded its own failure; stop the pipeline —
                    # later stages depend on this one's output
                    log.exception("stage %s failed", st)
                    return
        finally:
            conn.close()

    def _daily_scan_due() -> bool:
        conn = connect(config.db_path)
        try:
            enabled = get_meta(conn, "daily_scan", "off") == "on"
            row = conn.execute(
                "SELECT started_at FROM scans WHERE stage = 'sync' "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        last = None
        if row and row["started_at"]:
            try:
                last = datetime.fromisoformat(row["started_at"])
            except ValueError:
                last = None
        return scan_is_due(enabled, last, datetime.now(UTC))

    def _trigger_daily_scan() -> None:
        if auth_mode() is None:
            return  # nothing to scan until Drive is connected
        # start() no-ops if a job is already running
        app.state.runner.start("all", lambda: run_stage_job("all"))

    app.state.scheduler = None
    if enable_scheduler:
        app.state.scheduler = DailyScheduler(_daily_scan_due, _trigger_daily_scan)
        app.state.scheduler.start()

    @app.post("/schedule")
    def toggle_schedule(conn: sqlite3.Connection = Depends(get_conn)):
        """Turn the daily automatic full scan on or off."""
        current = get_meta(conn, "daily_scan", "off")
        set_meta(conn, "daily_scan", "off" if current == "on" else "on")
        return RedirectResponse("/", status_code=303)

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(
        request: Request,
        saved: int = 0,
        conn: sqlite3.Connection = Depends(get_conn),
    ):
        return templates.TemplateResponse(
            request,
            "settings.html",
            {
                "review_mode": get_meta(conn, "review_mode", "scroll"),
                "daily_enabled": get_meta(conn, "daily_scan", "off") == "on",
                "saved": saved,
            },
        )

    @app.post("/settings")
    async def save_settings(
        request: Request, conn: sqlite3.Connection = Depends(get_conn)
    ):
        form = await request.form()
        mode = str(form.get("review_mode", "scroll"))
        set_meta(conn, "review_mode", "all" if mode == "all" else "scroll")
        return RedirectResponse("/settings?saved=1", status_code=303)

    @app.get("/setup", response_class=HTMLResponse)
    def setup(
        request: Request,
        msg: str | None = None,
        err: str | None = None,
        ollama_host: str | None = None,
    ):
        host = ollama_host or config.ollama.host
        tested = ollama_host is not None  # arrived via the "test connection" form
        models: list[str] | None = None
        ollama_status: dict | None = None
        # only probe Ollama on an explicit test — a plain page load (incl. the
        # one `make run` auto-opens) must never block on a slow/absent host
        if tested:
            try:
                models = ollama_lister(host)
                if models:
                    ollama_status = {
                        "ok": True,
                        "kind": "ok",
                        "message": f"Test successful — {len(models)} usable "
                        f"model{'s' if len(models) != 1 else ''} found.",
                    }
                else:
                    ollama_status = {
                        "ok": False,
                        "kind": "no_models",
                        "message": "Connected, but no multi-image vision models "
                        "are installed yet.",
                    }
            except Exception as exc:
                ollama_status = {
                    "ok": False,
                    "kind": "unreachable",
                    "message": f"Test failed — could not reach Ollama at {host}.",
                    "detail": str(exc),
                }
        mode = auth_mode()
        return templates.TemplateResponse(
            request,
            "setup.html",
            {
                "msg": msg,
                "err": err,
                "auth_mode": mode,
                "authorized": mode is not None,
                "credentials_present": Path("credentials.json").exists(),
                "sa_email": service_account_email()
                if mode == "service_account"
                else None,
                "browse_root": "shared" if mode == "service_account" else "root",
                "host": host,
                "saved_host": config.ollama.host,
                "saved_model": config.ollama.model,
                "models": models,
                "ollama_status": ollama_status,
                "tested": tested,
                "folder_id": config.drive_folder_id,
            },
        )

    @app.get("/drive/browse", response_class=HTMLResponse)
    def drive_browse(request: Request, folder: str = "root", home: str = "root"):
        """Folder-picker partial: subfolders of `folder`, with breadcrumb and
        a scan-this-folder button. Drives the setup wizard's scope step.

        `home` is the top of this browse session — 'root' (My Drive) for OAuth,
        'shared' (folders shared with the service account) for service-account
        mode, where the account's own My Drive is empty."""
        try:
            client = drive_client_factory()
        except CredentialsRequired:
            return HTMLResponse(
                '<p class="muted">Connect Google Drive first (step 1).</p>'
            )
        try:
            if folder == "shared":
                meta = {"id": "shared", "name": "Folders shared with doppel"}
                subfolders = client.list_shared_folders()
                parent = None
            else:
                meta = client.get_folder(folder)
                subfolders = client.list_child_folders(folder)
                parents = meta.get("parents") or []
                parent = parents[0] if parents else None
        except Exception as exc:
            return HTMLResponse(
                f'<p class="error">could not read that folder: '
                f"{html.escape(str(exc))}</p>"
            )
        # 'root' is only an input alias — Drive returns My Drive's real id with
        # no parents, so detect the top of the session by parentlessness, not a
        # string compare against 'root' (which never matches the resolved id)
        is_top = folder == "shared" or parent is None
        return templates.TemplateResponse(
            request,
            "drive_browse.html",
            {
                "current": meta,
                "home": home,
                "is_top": is_top,
                # 'up' only makes sense in My-Drive mode (a shared folder's
                # real parent usually isn't accessible to the service account)
                "show_up": home == "root" and not is_top and parent is not None,
                "parent": parent,
                "can_scan_all": home == "root" and is_top,  # "entire Drive"
                "subfolders": sorted(subfolders, key=lambda f: f["name"].lower()),
                "saved_folder": config.drive_folder_id,
            },
        )

    @app.post("/setup/credentials")
    async def upload_credentials(request: Request):
        import json as _json
        from urllib.parse import quote

        form = await request.form()
        upload = form.get("credentials")
        if upload is None or not hasattr(upload, "read"):
            return RedirectResponse("/setup?err=no+file+uploaded", status_code=303)
        data = await upload.read()
        # a service-account key is the recommended, cleaner path — accept it
        # and route it to service_account.json
        if is_service_account_key(data):
            Path(SERVICE_ACCOUNT_PATH).write_bytes(data)
            email = service_account_email() or ""
            return RedirectResponse(
                "/setup?msg="
                + quote(
                    "Service account connected. Now share your photos folder in "
                    f"Google Drive with: {email}"
                ),
                status_code=303,
            )
        try:
            parsed = _json.loads(data)
        except ValueError:
            parsed = {}
        if "installed" in parsed or "web" in parsed:
            Path("credentials.json").write_bytes(data)
            return RedirectResponse(
                "/setup?msg=" + quote("OAuth client saved — now sign in"),
                status_code=303,
            )
        return RedirectResponse(
            "/setup?err="
            + quote(
                "That JSON isn't a Google credential — upload a service-account "
                "key (recommended) or a Desktop OAuth client from Cloud Console."
            ),
            status_code=303,
        )

    @app.post("/setup/disconnect")
    def disconnect_drive(request: Request):
        """Remove the active Drive credential so the user can switch modes or
        reconnect — otherwise a service-account key silently and permanently
        shadows an OAuth connection with no way back from the UI."""
        from urllib.parse import quote

        mode = auth_mode()
        if mode == "service_account":
            Path(SERVICE_ACCOUNT_PATH).unlink(missing_ok=True)
        elif mode == "oauth":
            Path("token.json").unlink(missing_ok=True)
        with init_lock:
            app.state.fetcher = None  # drop the client built from old creds
        return RedirectResponse(
            "/setup?msg=" + quote("Disconnected from Google Drive."),
            status_code=303,
        )

    @app.post("/oauth/start")
    def oauth_start(request: Request):
        # POST (not GET) so the same-origin guard covers it: a GET is
        # reachable by top-level navigation, which carries no Origin
        if not Path("credentials.json").exists():
            return RedirectResponse(
                "/setup?err=upload+credentials.json+first", status_code=303
            )
        redirect_uri = str(request.url_for("oauth_callback"))
        flow = oauth_flow_factory(redirect_uri)
        auth_url, state = flow.authorization_url(
            access_type="offline", prompt="consent"
        )
        app.state.oauth = {"state": state, "redirect_uri": redirect_uri}
        return RedirectResponse(auth_url, status_code=303)

    @app.get("/oauth/callback", name="oauth_callback")
    def oauth_callback(
        request: Request,
        state: str | None = None,
        code: str | None = None,
        error: str | None = None,
    ):
        if error:
            return RedirectResponse(
                f"/setup?err=authorization+failed:+{html.escape(error)}",
                status_code=303,
            )
        from urllib.parse import quote

        pending = app.state.oauth
        if not pending or not state or state != pending["state"] or not code:
            # state lost (server restarted, double-start, or forgery): send
            # the user back to the wizard to retry, not a raw 400 dead-end
            return RedirectResponse(
                "/setup?err=" + quote("authorization expired — click authorize again"),
                status_code=303,
            )
        flow = oauth_flow_factory(pending["redirect_uri"])
        try:
            flow.fetch_token(code=code)
            Path("token.json").write_text(flow.credentials.to_json())
        except Exception as exc:
            return RedirectResponse(
                "/setup?err=" + quote(f"token exchange failed: {exc}"),
                status_code=303,
            )
        app.state.oauth = None
        with init_lock:
            app.state.fetcher = None  # rebuild with the fresh credentials
        return RedirectResponse("/setup?msg=Google+Drive+authorized", status_code=303)

    @app.post("/setup/ollama")
    async def save_ollama(request: Request):
        from urllib.parse import quote

        form = await request.form()
        host = str(form.get("host", "")).strip()
        model = str(form.get("model", "")).strip()
        if not host or not model:
            return RedirectResponse(
                "/setup?err=host+and+model+are+required", status_code=303
            )
        try:
            models = ollama_lister(host)
        except Exception as exc:
            return RedirectResponse(
                f"/setup?err={quote(f'cannot reach Ollama at {host}: {exc}')}",
                status_code=303,
            )
        if model not in models:
            return RedirectResponse(
                f"/setup?err={quote(f'model {model} is not installed on {host}')}",
                status_code=303,
            )
        set_config_value(config_path, "host", host, section="ollama")
        set_config_value(config_path, "model", model, section="ollama")
        reload_config()
        return RedirectResponse(
            f"/setup?msg={quote(f'Ollama set to {model} at {host}')}",
            status_code=303,
        )

    @app.post("/setup/folder")
    async def save_folder(request: Request):
        from urllib.parse import quote

        form = await request.form()
        raw = str(form.get("folder", ""))
        try:
            folder_id = parse_folder_input(raw)
        except ValueError as exc:
            return RedirectResponse(f"/setup?err={quote(str(exc))}", status_code=303)
        label = "entire Drive"
        if folder_id is not None:
            try:
                meta = drive_client_factory().get_folder(folder_id)
            except CredentialsRequired:
                return RedirectResponse(
                    "/setup?err=authorize+Google+Drive+first+(step+1)",
                    status_code=303,
                )
            except Exception as exc:
                return RedirectResponse(
                    f"/setup?err={quote(f'folder not accessible: {exc}')}",
                    status_code=303,
                )
            label = f"folder “{meta['name']}”"
        set_config_value(config_path, "drive_folder_id", folder_id or "")
        reload_config()
        return RedirectResponse(
            f"/setup?msg={quote(f'scan scope set to {label}')}", status_code=303
        )

    def _left_pane_ctx(conn: sqlite3.Connection) -> dict:
        """Everything the persistent left pane shows: library totals, per-tier
        group counts (the tier navigation), pending trash, and scan status."""
        photo_count = conn.execute(
            "SELECT COUNT(*) AS n FROM photos WHERE status = 'active'"
        ).fetchone()["n"]
        tier_counts = {
            row["tier"]: row["n"]
            for row in conn.execute(
                "SELECT tier, COUNT(*) AS n FROM groups GROUP BY tier"
            )
        }
        # pending trash = photos still in Drive and marked trash (already-moved
        # ones are status='trashed' and shouldn't inflate the count)
        trash_count = conn.execute(
            "SELECT COUNT(*) AS n FROM decisions d JOIN photos p ON p.id = d.photo_id "
            "WHERE d.action = 'trash' AND p.status = 'active'"
        ).fetchone()["n"]
        reclaim_total = conn.execute(
            """
            SELECT COALESCE(SUM(p.size), 0) AS n FROM decisions d
            JOIN photos p ON p.id = d.photo_id
            WHERE d.action = 'trash' AND p.status = 'active'
            """
        ).fetchone()["n"]
        last_full = conn.execute(
            "SELECT finished_at FROM scans WHERE stage = 'sync' AND status = 'done' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        # a stage is "left interrupted" only if its MOST RECENT scan failed as
        # interrupted (a later clean run clears it) — that's when we offer resume
        interrupted = conn.execute(
            """
            SELECT s.stage FROM scans s
            JOIN (SELECT stage, MAX(id) AS mx FROM scans GROUP BY stage) latest
              ON latest.stage = s.stage AND latest.mx = s.id
            WHERE s.status = 'failed' AND s.error = 'interrupted'
            LIMIT 1
            """
        ).fetchone()
        return {
            "photo_count": photo_count,
            "tier_counts": tier_counts,
            "trash_count": trash_count,
            "reclaim_total": reclaim_total,
            "scans": scan_overview(conn),
            "any_running": app.state.runner.running_stage() is not None,
            "needs_setup": auth_mode() is None,
            "folder_id": config.drive_folder_id,
            "daily_enabled": get_meta(conn, "daily_scan", "off") == "on",
            "last_full_scan": last_full["finished_at"] if last_full else None,
            "interrupted_stage": interrupted["stage"] if interrupted else None,
        }

    def _review_pane_ctx(
        conn: sqlite3.Connection,
        tier: str,
        reviewed: str,
        sort: str,
        variants: bool,
    ) -> dict:
        """Context for the right review pane: the group cards for the current
        tier/filter/sort plus the controls' current state."""
        sort = resolve_sort(tier, sort)
        all_at_once = get_meta(conn, "review_mode", "scroll") == "all"
        ids, total, stats = _review_page_ids(
            conn,
            tier,
            reviewed,
            sort,
            variants,
            page=1,
            limit=None if all_at_once else REVIEW_BATCH,
        )
        groups = [_group_context(conn, gid) for gid in ids]
        reclaim_total = conn.execute(
            """
            SELECT COALESCE(SUM(p.size), 0) AS n FROM decisions d
            JOIN photos p ON p.id = d.photo_id
            WHERE d.action = 'trash' AND p.status = 'active'
            """
        ).fetchone()["n"]
        return {
            "tier": tier,
            "reviewed": reviewed,
            "sort": sort,
            "variants": variants,
            "sort_options": sort_options(tier),
            "groups": groups,
            "has_more": (not all_at_once) and total > REVIEW_BATCH,
            "next_page": 2,
            "total": total,
            "stats": stats,
            "reclaim_total": reclaim_total,
        }

    def _render_workspace(
        request: Request,
        conn: sqlite3.Connection,
        tier: str | None,
        reviewed: str,
        sort: str | None,
        variants: str,
    ) -> HTMLResponse:
        ctx = _left_pane_ctx(conn)
        ctx["active_tier"] = tier
        if tier:
            # merge the review-pane vars to the top level so the shared
            # _review_pane.html partial reads the same names here and when the
            # /review/pane route returns it standalone
            ctx.update(
                _review_pane_ctx(conn, tier, reviewed, sort or "", _truthy(variants))
            )
        return templates.TemplateResponse(request, "workspace.html", ctx)

    @app.get("/", response_class=HTMLResponse)
    def home(
        request: Request,
        tier: str | None = None,
        reviewed: str = "all",
        sort: str | None = None,
        variants: str = "",
        conn: sqlite3.Connection = Depends(get_conn),
    ):
        return _render_workspace(request, conn, tier, reviewed, sort, variants)

    @app.get("/review", response_class=HTMLResponse)
    def review(
        request: Request,
        tier: str = "exact",
        reviewed: str = "all",
        sort: str | None = None,
        variants: str = "",
        conn: sqlite3.Connection = Depends(get_conn),
    ):
        return _render_workspace(request, conn, tier, reviewed, sort, variants)

    @app.get("/review/pane", response_class=HTMLResponse)
    def review_pane(
        request: Request,
        tier: str = "exact",
        reviewed: str = "all",
        sort: str | None = None,
        variants: str = "",
        conn: sqlite3.Connection = Depends(get_conn),
    ):
        ctx = _review_pane_ctx(conn, tier, reviewed, sort or "", _truthy(variants))
        resp = templates.TemplateResponse(request, "_review_pane.html", ctx)
        # keep the address bar in sync so refresh/bookmark/share preserve the
        # tier + filter + sort the user is actually looking at
        resp.headers["HX-Push-Url"] = _pane_push_url(
            tier, ctx["reviewed"], ctx["sort"], ctx["variants"]
        )
        return resp

    @app.get("/partials/scans", response_class=HTMLResponse)
    def scans_partial(request: Request, conn: sqlite3.Connection = Depends(get_conn)):
        # reuse the whole left-pane context so the 2s poll can also push live
        # library/tier/trash counts out-of-band (oob=True) — results the scan
        # is finding show up in the nav in real time
        ctx = _left_pane_ctx(conn)
        ctx["oob"] = True
        return templates.TemplateResponse(request, "_scan_status.html", ctx)

    @app.post("/scans/{stage}", response_class=HTMLResponse)
    def start_stage(
        request: Request, stage: str, conn: sqlite3.Connection = Depends(get_conn)
    ):
        error = None
        if stage != "all" and stage not in UI_STAGES:
            raise HTTPException(status_code=404, detail=f"unknown stage {stage!r}")
        if stage in ("sync", "all") and auth_mode() is None:
            error = "Google Drive is not connected yet — open the setup wizard."
        elif not app.state.runner.start(stage, lambda: run_stage_job(stage)):
            error = "a job is already running"
        # the error goes into #scan-error via an out-of-band swap: the polled
        # #scan-status region is replaced every 2s, which would erase it
        table = templates.get_template("_scan_status.html").render(
            request=request,
            scans=scan_overview(conn),
            any_running=app.state.runner.running_stage() is not None,
        )
        message = f'<p class="error">{html.escape(error)}</p>' if error else ""
        oob = f'<div id="scan-error" hx-swap-oob="true">{message}</div>'
        return HTMLResponse(table + oob)

    @app.get("/scans/{scan_id}")
    def scan_status(scan_id: int, conn: sqlite3.Connection = Depends(get_conn)):
        row = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="no such scan")
        return dict(row)

    def _group_context(conn: sqlite3.Connection, group_id: int) -> dict | None:
        """Everything a group card/detail needs: members (largest first), the
        keep/trash selection (largest kept by default, saved decisions win),
        whether it's fully reviewed, reclaimable bytes, and VLM verdicts."""
        group = conn.execute(
            "SELECT * FROM groups WHERE id = ?", (group_id,)
        ).fetchone()
        if group is None:
            return None
        members = conn.execute(
            """
            SELECT p.*, m.score FROM group_members m
            JOIN photos p ON p.id = m.photo_id
            WHERE m.group_id = ? ORDER BY p.size DESC
            """,
            (group_id,),
        ).fetchall()
        decisions = {
            row["photo_id"]: row["action"]
            for row in conn.execute(
                """
                SELECT d.photo_id, d.action FROM decisions d
                JOIN group_members m ON m.photo_id = d.photo_id
                WHERE m.group_id = ?
                """,
                (group_id,),
            )
        }
        default = default_selection(
            members, config.prefer_trash_sort, config.sort_folder_keyword
        )
        selected = {p["id"]: decisions.get(p["id"], default[p["id"]]) for p in members}
        reviewed = len(decisions) == len(members) and len(members) > 0
        reclaim = sum((p["size"] or 0) for p in members if selected[p["id"]] == "trash")
        verdicts = []
        if group["tier"] == "vlm" and members:
            import json as _json

            names = {p["id"]: p["name"] for p in members}
            ids = list(names)
            placeholders = ",".join("?" * len(ids))
            seen_pairs: set[tuple[int, int]] = set()
            for row in conn.execute(
                f"""
                SELECT * FROM vlm_results
                WHERE task = 'adjudicate'
                  AND photo_id IN ({placeholders})
                  AND photo_id_b IN ({placeholders})
                ORDER BY id DESC
                """,  # noqa: S608 — placeholders only
                (*ids, *ids),
            ):
                pair = (row["photo_id"], row["photo_id_b"])
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                try:
                    reason = _json.loads(row["response"]).get("reason", "")
                except ValueError:
                    reason = ""
                verdicts.append(
                    {
                        "a": names[row["photo_id"]],
                        "b": names[row["photo_id_b"]],
                        "verdict": row["verdict"],
                        "reason": reason,
                    }
                )
        return {
            "group": group,
            "members": members,
            "selected": selected,
            "reviewed": reviewed,
            "reclaim": reclaim,
            "verdicts": verdicts,
            "confidence": group_confidence(
                group["tier"], [m["score"] for m in members]
            ),
        }

    def _group_card_response(
        request: Request, conn: sqlite3.Connection, group_id: int
    ) -> HTMLResponse:
        """Render a group card, or a graceful placeholder if the group vanished
        (a concurrent scan rebuild can delete it between the write and here)."""
        ctx = _group_context(conn, group_id)
        if ctx is None:
            return HTMLResponse(
                f'<div class="review-group" id="group-{group_id}">'
                '<p class="muted">This group changed during a scan — '
                "reload to see the latest.</p></div>"
            )
        return templates.TemplateResponse(request, "_group_card.html", {"g": ctx})

    def _review_page_ids(
        conn: sqlite3.Connection,
        tier: str,
        reviewed: str,
        sort: str = "confidence",
        variants: bool = False,
        page: int = 1,
        limit: int | None = REVIEW_BATCH,
    ) -> tuple[list[int], int, dict]:
        """Group ids for a review batch (or all groups when limit is None),
        the total group count, and review stats. Ordering comes from the
        validated `sort` key; `variants` restricts to color-variant groups."""
        having = REVIEWED_FILTERS.get(reviewed)
        if having is None:
            raise HTTPException(status_code=422, detail="reviewed must be all|yes|no")
        variant_clause = "AND g.color_variant = 1" if variants else ""
        base_query = f"""
            SELECT g.id, COUNT(m.photo_id) AS members,
                   COUNT(d.photo_id) AS decided,
                   MIN(m.score) AS min_score, MAX(m.score) AS max_score
            FROM groups g
            JOIN group_members m ON m.group_id = g.id
            JOIN photos p2 ON p2.id = m.photo_id
            LEFT JOIN decisions d ON d.photo_id = m.photo_id
            WHERE g.tier = ? {variant_clause}
            GROUP BY g.id {having}
        """  # noqa: S608 — `having`/`variant_clause` from fixed maps above
        total = conn.execute(
            f"SELECT COUNT(*) AS n FROM ({base_query})", (tier,)
        ).fetchone()["n"]
        order = sort_order_clause(tier, resolve_sort(tier, sort))
        if limit is None:
            ids = [
                r["id"]
                for r in conn.execute(f"{base_query} {order}", (tier,))  # noqa: S608
            ]
        else:
            ids = [
                r["id"]
                for r in conn.execute(
                    f"{base_query} {order} LIMIT ? OFFSET ?",  # noqa: S608
                    (tier, limit, (page - 1) * limit),
                )
            ]
        # tier-wide stats (independent of the reviewed filter)
        stats = conn.execute(
            """
            SELECT COUNT(*) AS total_groups,
                   SUM(CASE WHEN decided = members THEN 1 ELSE 0 END) AS reviewed_groups
            FROM (
              SELECT g.id, COUNT(m.photo_id) AS members, COUNT(d.photo_id) AS decided
              FROM groups g JOIN group_members m ON m.group_id = g.id
              LEFT JOIN decisions d ON d.photo_id = m.photo_id
              WHERE g.tier = ? GROUP BY g.id
            )
            """,
            (tier,),
        ).fetchone()
        return ids, total, dict(stats)

    @app.get("/review/groups", response_class=HTMLResponse)
    def review_groups(
        request: Request,
        tier: str = "exact",
        reviewed: str = "all",
        sort: str | None = None,
        variants: str = "",
        page: int = 2,
        conn: sqlite3.Connection = Depends(get_conn),
    ):
        sort = resolve_sort(tier, sort)
        variants_on = _truthy(variants)
        ids, total, _ = _review_page_ids(
            conn, tier, reviewed, sort, variants_on, page=page
        )
        groups = [_group_context(conn, gid) for gid in ids]
        return templates.TemplateResponse(
            request,
            "_review_batch.html",
            {
                "tier": tier,
                "reviewed": reviewed,
                "sort": sort,
                # a bool, so the next-page sentinel agrees with page 1 no matter
                # how the incoming param was spelled (avoids skip/dup across pages)
                "variants": variants_on,
                "groups": groups,
                "has_more": page * REVIEW_BATCH < total,
                "next_page": page + 1,
            },
        )

    @app.get("/groups", response_class=HTMLResponse)
    def group_list(
        request: Request,
        tier: str = "exact",
        page: int = 1,
        reviewed: str = "all",
        conn: sqlite3.Connection = Depends(get_conn),
    ):
        having = {
            "all": "",
            "yes": "HAVING decided = members",
            "no": "HAVING decided < members",
        }.get(reviewed)
        if having is None:
            raise HTTPException(status_code=422, detail="reviewed must be all|yes|no")
        base_query = f"""
            SELECT g.id, g.tier, g.color_variant,
                   COUNT(m.photo_id) AS members,
                   COUNT(d.photo_id) AS decided
            FROM groups g
            JOIN group_members m ON m.group_id = g.id
            LEFT JOIN decisions d ON d.photo_id = m.photo_id
            WHERE g.tier = ?
            GROUP BY g.id {having}
        """  # noqa: S608 — `having` comes from the fixed map above
        total = conn.execute(
            f"SELECT COUNT(*) AS n FROM ({base_query})", (tier,)
        ).fetchone()["n"]
        groups = conn.execute(
            f"{base_query} ORDER BY g.id LIMIT ? OFFSET ?",
            (tier, PAGE_SIZE, (page - 1) * PAGE_SIZE),
        ).fetchall()
        strips = {
            g["id"]: conn.execute(
                """
                SELECT p.id FROM group_members m JOIN photos p ON p.id = m.photo_id
                WHERE m.group_id = ? ORDER BY p.size DESC LIMIT 4
                """,
                (g["id"],),
            ).fetchall()
            for g in groups
        }
        return templates.TemplateResponse(
            request,
            "groups.html",
            {
                "tier": tier,
                "groups": groups,
                "strips": strips,
                "page": page,
                "pages": max(1, -(-total // PAGE_SIZE)),
                "total": total,
                "reviewed": reviewed,
            },
        )

    @app.get("/groups/{group_id}", response_class=HTMLResponse)
    def group_detail(
        request: Request,
        group_id: int,
        conn: sqlite3.Connection = Depends(get_conn),
    ):
        group = conn.execute(
            "SELECT * FROM groups WHERE id = ?", (group_id,)
        ).fetchone()
        if group is None:
            raise HTTPException(status_code=404, detail="no such group")
        members = conn.execute(
            """
            SELECT p.*, m.score FROM group_members m
            JOIN photos p ON p.id = m.photo_id
            WHERE m.group_id = ? ORDER BY p.size DESC
            """,
            (group_id,),
        ).fetchall()
        decisions = {
            row["photo_id"]: row["action"]
            for row in conn.execute(
                """
                SELECT d.photo_id, d.action FROM decisions d
                JOIN group_members m ON m.photo_id = d.photo_id
                WHERE m.group_id = ?
                """,
                (group_id,),
            )
        }
        # default preselect: keep one copy (a real folder over a "sort" inbox,
        # else the largest), trash the rest; the user's saved decisions override
        default = default_selection(
            members, config.prefer_trash_sort, config.sort_folder_keyword
        )
        selected = {p["id"]: decisions.get(p["id"], default[p["id"]]) for p in members}
        verdicts = []
        if group["tier"] == "vlm" and members:
            import json as _json

            names = {p["id"]: p["name"] for p in members}
            ids = list(names)
            placeholders = ",".join("?" * len(ids))
            seen_pairs: set[tuple[int, int]] = set()
            for row in conn.execute(
                f"""
                SELECT * FROM vlm_results
                WHERE task = 'adjudicate'
                  AND photo_id IN ({placeholders})
                  AND photo_id_b IN ({placeholders})
                ORDER BY id DESC
                """,  # noqa: S608 — placeholders only
                (*ids, *ids),
            ):
                pair = (row["photo_id"], row["photo_id_b"])
                if pair in seen_pairs:
                    continue  # newest ruling per pair wins
                seen_pairs.add(pair)
                try:
                    reason = _json.loads(row["response"]).get("reason", "")
                except ValueError:
                    reason = ""
                verdicts.append(
                    {
                        "a": names[row["photo_id"]],
                        "b": names[row["photo_id_b"]],
                        "verdict": row["verdict"],
                        "reason": reason,
                        "model": row["model"],
                        "prompt_version": row["prompt_version"],
                    }
                )
        return templates.TemplateResponse(
            request,
            "group_detail.html",
            {
                "group": group,
                "members": members,
                "selected": selected,
                "verdicts": verdicts,
            },
        )

    @app.post("/groups/{group_id}/decisions")
    async def save_decisions(
        request: Request,
        group_id: int,
        conn: sqlite3.Connection = Depends(get_conn),
    ):
        member_ids = {
            row["photo_id"]
            for row in conn.execute(
                "SELECT photo_id FROM group_members WHERE group_id = ?",
                (group_id,),
            )
        }
        if not member_ids:
            # a concurrent scan rebuilt this tier between loading the card and
            # saving — degrade to the same graceful placeholder as a post-write
            # disappearance, rather than a raw 404
            if request.headers.get("HX-Request"):
                return _group_card_response(request, conn, group_id)
            return RedirectResponse(url="/review?tier=exact", status_code=303)
        form = await request.form()
        for key, action in form.items():
            if not key.startswith("action_"):
                continue
            if action not in ("keep", "trash"):
                raise HTTPException(
                    status_code=422, detail=f"invalid action {action!r}"
                )
            try:
                photo_id = int(key.removeprefix("action_"))
            except ValueError:
                continue  # malformed field name: ignore like foreign fields
            if photo_id not in member_ids:
                continue  # stale or foreign field: ignore
            conn.execute(
                """
                INSERT INTO decisions (photo_id, action, decided_at)
                VALUES (?, ?, ?)
                ON CONFLICT(photo_id) DO UPDATE SET
                  action = excluded.action, decided_at = excluded.decided_at
                """,
                (photo_id, action, now()),
            )
        conn.commit()
        # htmx (the one-page review) swaps the updated card in place; a plain
        # form submit (the standalone detail page) redirects as before
        if request.headers.get("HX-Request"):
            return _group_card_response(request, conn, group_id)
        return RedirectResponse(url=f"/groups/{group_id}", status_code=303)

    @app.post("/groups/{group_id}/keep", response_class=HTMLResponse)
    def keep_group(
        request: Request,
        group_id: int,
        conn: sqlite3.Connection = Depends(get_conn),
    ):
        """'Keep group' — these aren't duplicates: mark every member keep."""
        member_ids = [
            row["photo_id"]
            for row in conn.execute(
                "SELECT photo_id FROM group_members WHERE group_id = ?", (group_id,)
            )
        ]
        if not member_ids:
            # group gone (e.g. a concurrent scan rebuilt the tier) — the same
            # graceful placeholder the card falls back to elsewhere
            return _group_card_response(request, conn, group_id)
        for pid in member_ids:
            conn.execute(
                """
                INSERT INTO decisions (photo_id, action, decided_at)
                VALUES (?, 'keep', ?)
                ON CONFLICT(photo_id) DO UPDATE SET
                  action = 'keep', decided_at = excluded.decided_at
                """,
                (pid, now()),
            )
        conn.commit()
        return _group_card_response(request, conn, group_id)

    @app.post("/review/auto")
    def auto_resolve(
        request: Request,
        tier: str = "exact",
        reviewed: str = "no",
        sort: str | None = None,
        variants: str = "",
        conn: sqlite3.Connection = Depends(get_conn),
    ):
        """Auto-resolve every UNTOUCHED group in the tier: keep the largest
        file, trash the rest. Any group you've started deciding — even a single
        keep/trash — is left completely alone, so this never overwrites your
        manual choices."""
        group_ids = [
            r["id"]
            for r in conn.execute(
                """
                SELECT g.id FROM groups g
                JOIN group_members m ON m.group_id = g.id
                LEFT JOIN decisions d ON d.photo_id = m.photo_id
                WHERE g.tier = ?
                GROUP BY g.id HAVING COUNT(d.photo_id) = 0
                """,
                (tier,),
            )
        ]
        for gid in group_ids:
            members = conn.execute(
                """
                SELECT p.id, p.size, p.folder_path
                FROM group_members m JOIN photos p ON p.id = m.photo_id
                WHERE m.group_id = ? ORDER BY p.size DESC
                """,
                (gid,),
            ).fetchall()
            default = default_selection(
                members, config.prefer_trash_sort, config.sort_folder_keyword
            )
            for row in members:
                conn.execute(
                    """
                    INSERT INTO decisions (photo_id, action, decided_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(photo_id) DO UPDATE SET
                      action = excluded.action, decided_at = excluded.decided_at
                    """,
                    (row["id"], default[row["id"]], now()),
                )
        conn.commit()
        # htmx (the workspace) swaps the refreshed review pane in place; a plain
        # POST falls back to a full navigation
        if request.headers.get("HX-Request"):
            ctx = _review_pane_ctx(conn, tier, reviewed, sort or "", _truthy(variants))
            resp = templates.TemplateResponse(request, "_review_pane.html", ctx)
            resp.headers["HX-Push-Url"] = _pane_push_url(
                tier, ctx["reviewed"], ctx["sort"], ctx["variants"]
            )
            return resp
        return RedirectResponse(
            url=f"/review?tier={tier}&reviewed={reviewed}", status_code=303
        )

    def _finalize_group(conn: sqlite3.Connection, ctx: dict) -> None:
        """Commit a group's shown keep/trash selection as decisions, so every
        member is decided and the group reads as reviewed. Idempotent."""
        for p in ctx["members"]:
            conn.execute(
                "INSERT INTO decisions (photo_id, action, decided_at) "
                "VALUES (?, ?, ?) ON CONFLICT(photo_id) DO UPDATE SET "
                "action = excluded.action, decided_at = excluded.decided_at",
                (p["id"], ctx["selected"][p["id"]], now()),
            )

    def _pane_response(
        request: Request,
        conn: sqlite3.Connection,
        tier: str,
        reviewed: str,
        sort: str | None,
        variants: str,
    ):
        """Re-render the review pane after a bulk action (htmx swap), or fall
        back to a full navigation for a plain POST."""
        if request.headers.get("HX-Request"):
            ctx = _review_pane_ctx(conn, tier, reviewed, sort or "", _truthy(variants))
            resp = templates.TemplateResponse(request, "_review_pane.html", ctx)
            resp.headers["HX-Push-Url"] = _pane_push_url(
                tier, ctx["reviewed"], ctx["sort"], ctx["variants"]
            )
            return resp
        return RedirectResponse(
            url=f"/review?tier={tier}&reviewed={reviewed}", status_code=303
        )

    @app.post("/groups/{group_id}/reviewed", response_class=HTMLResponse)
    def set_group_reviewed(
        request: Request,
        group_id: int,
        value: str = "1",
        conn: sqlite3.Connection = Depends(get_conn),
    ):
        """Toggle a group's Reviewed state. 'Reviewed' rides on the decisions
        table because group ids are rebuilt on every scan: checking commits the
        group's shown keep/trash selection (every member decided); unchecking
        clears those decisions."""
        ctx = _group_context(conn, group_id)
        if ctx is None:
            return _group_card_response(request, conn, group_id)
        if value == "1":
            _finalize_group(conn, ctx)
        else:
            ids = [p["id"] for p in ctx["members"]]
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"DELETE FROM decisions WHERE photo_id IN ({placeholders})",  # noqa: S608
                    ids,
                )
        conn.commit()
        return _group_card_response(request, conn, group_id)

    @app.post("/review/reviewed-all")
    def review_all(
        request: Request,
        tier: str = "exact",
        reviewed: str = "all",
        sort: str | None = None,
        variants: str = "",
        conn: sqlite3.Connection = Depends(get_conn),
    ):
        """Mark every group in the tier reviewed: commit each group's shown
        selection, filling undecided members with the default keep/trash choice
        without overwriting a manual one."""
        for row in conn.execute("SELECT id FROM groups WHERE tier = ?", (tier,)):
            ctx = _group_context(conn, row["id"])
            if ctx is not None:
                _finalize_group(conn, ctx)
        conn.commit()
        return _pane_response(request, conn, tier, reviewed, sort, variants)

    @app.post("/review/unreviewed-all")
    def unreview_all(
        request: Request,
        tier: str = "exact",
        reviewed: str = "all",
        sort: str | None = None,
        variants: str = "",
        conn: sqlite3.Connection = Depends(get_conn),
    ):
        """Clear every decision in the tier, so no group reads as reviewed."""
        conn.execute(
            "DELETE FROM decisions WHERE photo_id IN ("
            "  SELECT m.photo_id FROM group_members m "
            "  JOIN groups g ON g.id = m.group_id WHERE g.tier = ?)",
            (tier,),
        )
        conn.commit()
        return _pane_response(request, conn, tier, reviewed, sort, variants)

    def _pending_trash(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        """Photos marked trash whose groups are fully reviewed and that are still
        live in Drive (largest first, so the confirm list leads with the biggest
        space wins). A photo is excluded while ANY group it belongs to still has
        an undecided member — Move-to-Trash only acts on reviewed groups. Photos
        in no group (e.g. a direct decision) are always eligible."""
        return conn.execute(
            """
            SELECT p.id, p.drive_id, p.name, p.size, p.folder_path
            FROM decisions d JOIN photos p ON p.id = d.photo_id
            WHERE d.action = 'trash' AND p.status = 'active'
              AND NOT EXISTS (
                SELECT 1 FROM group_members m
                JOIN group_members m2 ON m2.group_id = m.group_id
                LEFT JOIN decisions d2 ON d2.photo_id = m2.photo_id
                WHERE m.photo_id = p.id AND d2.photo_id IS NULL
              )
            ORDER BY p.size DESC, p.name
            """
        ).fetchall()

    @app.get("/trash/confirm", response_class=HTMLResponse)
    def trash_confirm(request: Request, conn: sqlite3.Connection = Depends(get_conn)):
        items = _pending_trash(conn)
        return templates.TemplateResponse(
            request,
            "trash_confirm.html",
            {
                "items": items,
                "count": len(items),
                "total": sum((r["size"] or 0) for r in items),
                "can_trash": can_trash(),
                "auth": auth_mode(),
            },
        )

    @app.post("/trash")
    def do_trash(request: Request, conn: sqlite3.Connection = Depends(get_conn)):
        """Move every photo marked trash into Google Drive Trash — reversible,
        recoverable for 30 days, never a permanent delete. Reached only from the
        confirm page; each file is trashed independently so one failure (e.g. a
        read-only share) doesn't abort the rest."""
        items = _pending_trash(conn)
        if not items:
            return RedirectResponse("/review?tier=exact", status_code=303)
        try:
            client = trash_client_factory()
        except (CredentialsRequired, TrashNotAuthorized) as exc:
            return templates.TemplateResponse(
                request,
                "trash_confirm.html",
                {
                    "items": items,
                    "count": len(items),
                    "total": sum((r["size"] or 0) for r in items),
                    "can_trash": False,
                    "auth": auth_mode(),
                    "error": str(exc),
                },
                status_code=403,
            )
        moved, freed, failures = 0, 0, []
        for row in items:
            try:
                client.trash_file(row["drive_id"])
            except Exception as exc:  # noqa: BLE001 — surfaced per-file to the user
                failures.append({"name": row["name"], "error": str(exc)})
                continue
            conn.execute(
                "UPDATE photos SET status = 'trashed' WHERE id = ?", (row["id"],)
            )
            moved += 1
            freed += row["size"] or 0
        conn.commit()
        return templates.TemplateResponse(
            request,
            "trash_result.html",
            {"moved": moved, "freed": freed, "failures": failures},
        )

    @app.get("/export")
    def export_csv(conn: sqlite3.Connection = Depends(get_conn)):
        import csv
        import io

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["drive_id", "name", "size", "md5", "url"])
        for row in conn.execute(
            """
            SELECT p.drive_id, p.name, p.size, p.md5 FROM decisions d
            JOIN photos p ON p.id = d.photo_id
            WHERE d.action = 'trash' AND p.status = 'active' ORDER BY p.name
            """
        ):
            writer.writerow(
                [
                    row["drive_id"],
                    row["name"],
                    row["size"],
                    row["md5"],
                    f"https://drive.google.com/file/d/{row['drive_id']}/view",
                ]
            )
        return Response(
            content=buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=doppel-trash.csv"},
        )

    @app.get("/thumb/{photo_id}")
    def thumb(
        photo_id: int,
        conn: sqlite3.Connection = Depends(get_conn),
        fetcher: ImageFetcher = Depends(get_fetcher),
    ):
        row = conn.execute(
            "SELECT drive_id FROM photos WHERE id = ?", (photo_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="no such photo")
        try:
            path = fetcher.get(row["drive_id"], config.thumb_size)
        except FetchError as exc:
            # transient under heavy review-while-scanning load (rate limit /
            # blip). Logged so a burst of these is visible in logs/, not a 502
            # that just looks like a broken image.
            log.warning("thumb %s fetch failed: %s", photo_id, exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return FileResponse(path, media_type="image/jpeg")

    return app


def build() -> FastAPI:
    """uvicorn factory target (see Makefile run). Real runs get the daily-scan
    scheduler and on-disk diagnostics (rotating app log + native-crash dumps in
    logs/); tests construct create_app() directly without either."""
    from doppel.logsetup import setup_diagnostics

    setup_diagnostics(app_name="doppel")
    return create_app(enable_scheduler=True)
