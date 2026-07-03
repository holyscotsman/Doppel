"""FastAPI app: routes, templates, job wiring. Server-rendered UI (Jinja2 +
htmx); binds to 127.0.0.1 only (see Makefile run target)."""

from __future__ import annotations

import html
import sqlite3
import threading
from collections.abc import Callable, Iterator
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
from doppel.db import connect
from doppel.drive import (
    SERVICE_ACCOUNT_PATH,
    CredentialsRequired,
    FetchError,
    GoogleDriveClient,
    ImageFetcher,
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
from doppel.stages.brand import correct_brand, run_brand
from doppel.stages.exact import run_exact
from doppel.stages.near import run_near
from doppel.stages.similar import run_similar
from doppel.vlm import OllamaClient, VlmClient

PACKAGE_DIR = Path(__file__).parent

# stages the UI can launch, in pipeline order; extended phase by phase
UI_STAGES = ["sync", "exact", "near", "similar", "adjudicate", "brand"]

PAGE_SIZE = 20


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
    return DriveImageFetcher(
        db_path=config.db_path,
        client=GoogleDriveClient(creds),
        session=AuthorizedSession(creds),
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
) -> FastAPI:
    """Build the app. Tests inject a Config and fake factories."""
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

    app = FastAPI(title="doppel")
    app.state.config = config
    app.state.runner = JobRunner()
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
        """Latest scans row per UI stage, plus whether it is the live job."""
        running = app.state.runner.running_stage()
        overview = []
        for stage in UI_STAGES:
            row = conn.execute(
                "SELECT * FROM scans WHERE stage = ? ORDER BY id DESC LIMIT 1",
                (stage,),
            ).fetchone()
            overview.append(
                {
                    "stage": stage,
                    "scan": dict(row) if row else None,
                    "is_running": stage == running,
                }
            )
        return overview

    def run_stage_job(stage: str) -> None:
        conn = connect(config.db_path)
        try:
            # acquire stage dependencies first; a failure here (e.g. missing
            # OAuth credentials) happens before the stage's own start_scan,
            # so record it in the ledger — otherwise the click would look
            # like a silent no-op in the UI. The interactive OAuth flow is
            # never run on this worker thread: it would block the runner
            # forever if the consent tab is missed.
            try:
                job: Callable[[], object]
                if stage == "sync":
                    client = drive_client_factory()
                    folder_ids = _resolve_scan_roots(client, config)
                    job = lambda: run_sync(  # noqa: E731
                        conn, client, config.cache_dir, folder_ids
                    )
                elif stage == "exact":
                    job = lambda: run_exact(conn)  # noqa: E731
                elif stage == "near":
                    fetcher = get_fetcher()
                    job = lambda: run_near(conn, fetcher, config)  # noqa: E731
                elif stage == "similar":
                    fetcher, embedder = get_fetcher(), get_embedder()
                    job = lambda: run_similar(conn, fetcher, embedder, config)  # noqa: E731
                elif stage == "adjudicate":
                    fetcher, vlm = get_fetcher(), get_vlm()
                    job = lambda: run_adjudicate(conn, fetcher, vlm, config)  # noqa: E731
                elif stage == "brand":
                    fetcher, vlm = get_fetcher(), get_vlm()
                    job = lambda: run_brand(conn, fetcher, vlm, config)  # noqa: E731
                else:
                    return
            except Exception as exc:
                scan_id = start_scan(conn, stage)
                detail = exc.detail if isinstance(exc, HTTPException) else exc
                fail_scan(conn, scan_id, f"{type(exc).__name__}: {detail}")
                return
            job()
        finally:
            conn.close()

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

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, conn: sqlite3.Connection = Depends(get_conn)):
        photo_count = conn.execute(
            "SELECT COUNT(*) AS n FROM photos WHERE status = 'active'"
        ).fetchone()["n"]
        tier_counts = {
            row["tier"]: row["n"]
            for row in conn.execute(
                "SELECT tier, COUNT(*) AS n FROM groups GROUP BY tier"
            )
        }
        trash_count = conn.execute(
            "SELECT COUNT(*) AS n FROM decisions WHERE action = 'trash'"
        ).fetchone()["n"]
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "photo_count": photo_count,
                "tier_counts": tier_counts,
                "trash_count": trash_count,
                "scans": scan_overview(conn),
                "needs_setup": auth_mode() is None,
                "folder_id": config.drive_folder_id,
            },
        )

    @app.get("/partials/scans", response_class=HTMLResponse)
    def scans_partial(request: Request, conn: sqlite3.Connection = Depends(get_conn)):
        return templates.TemplateResponse(
            request, "_scan_status.html", {"scans": scan_overview(conn)}
        )

    @app.post("/scans/{stage}", response_class=HTMLResponse)
    def start_stage(
        request: Request, stage: str, conn: sqlite3.Connection = Depends(get_conn)
    ):
        error = None
        if stage not in UI_STAGES:
            raise HTTPException(status_code=404, detail=f"unknown stage {stage!r}")
        if stage == "sync" and auth_mode() is None:
            error = "Google Drive is not connected yet — open the setup wizard."
        elif not app.state.runner.start(stage, lambda: run_stage_job(stage)):
            error = "a job is already running"
        # the error goes into #scan-error via an out-of-band swap: the polled
        # #scan-status region is replaced every 2s, which would erase it
        table = templates.get_template("_scan_status.html").render(
            request=request, scans=scan_overview(conn)
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
        # default preselect: keep the largest file (first row), trash the rest;
        # the user's saved decisions override
        selected = {
            p["id"]: decisions.get(p["id"], "keep" if i == 0 else "trash")
            for i, p in enumerate(members)
        }
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
            raise HTTPException(status_code=404, detail="no such group")
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
        return RedirectResponse(url=f"/groups/{group_id}", status_code=303)

    @app.get("/brands", response_class=HTMLResponse)
    def brand_summary(request: Request, conn: sqlite3.Connection = Depends(get_conn)):
        threshold = config.ollama.brand_review_max_confidence
        brands = conn.execute(
            """
            SELECT value, COUNT(*) AS n FROM tags
            WHERE kind = 'brand' GROUP BY value ORDER BY n DESC, value
            """
        ).fetchall()
        queue_count = conn.execute(
            """
            SELECT COUNT(*) AS n FROM tags
            WHERE kind = 'brand' AND source = 'vlm'
              AND (confidence IS NULL OR confidence <= ?)
            """,
            (threshold,),
        ).fetchone()["n"]
        return templates.TemplateResponse(
            request,
            "brands.html",
            {"brands": brands, "queue_count": queue_count, "threshold": threshold},
        )

    @app.get("/brands/photos", response_class=HTMLResponse)
    def brand_photos(
        request: Request,
        brand: str | None = None,
        queue: int = 0,
        conn: sqlite3.Connection = Depends(get_conn),
    ):
        threshold = config.ollama.brand_review_max_confidence
        clauses = ["t.kind = 'brand'"]
        params: list = []
        if brand is not None:
            clauses.append("t.value = ?")
            params.append(brand)
        if queue:
            clauses.append(
                "t.source = 'vlm' AND (t.confidence IS NULL OR t.confidence <= ?)"
            )
            params.append(threshold)
        photos = conn.execute(
            f"""
            SELECT p.id, p.name, t.value, t.confidence, t.source,
                   (SELECT v.response FROM vlm_results v
                    WHERE v.task = 'brand' AND v.photo_id = p.id
                    ORDER BY v.id DESC LIMIT 1) AS response
            FROM tags t JOIN photos p ON p.id = t.photo_id
            WHERE {" AND ".join(clauses)}
            ORDER BY t.confidence, p.name
            """,  # noqa: S608 — clauses are fixed strings, values are bound
            params,
        ).fetchall()
        rows = []
        for p in photos:
            evidence = ""
            if p["response"]:
                import json as _json

                try:
                    evidence = _json.loads(p["response"]).get("evidence", "")
                except ValueError:
                    pass
            rows.append({**dict(p), "evidence": evidence})
        return templates.TemplateResponse(
            request,
            "brand_photos.html",
            {"photos": rows, "brand": brand, "queue": queue},
        )

    @app.post("/photos/{photo_id}/brand")
    async def save_brand_correction(
        request: Request,
        photo_id: int,
        conn: sqlite3.Connection = Depends(get_conn),
    ):
        row = conn.execute("SELECT id FROM photos WHERE id = ?", (photo_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="no such photo")
        form = await request.form()
        value = str(form.get("value", "")).strip()
        if not value:
            raise HTTPException(status_code=422, detail="brand value required")
        correct_brand(conn, photo_id, value)
        from urllib.parse import urlencode

        params: dict[str, str] = {}
        if form.get("brand"):
            params["brand"] = str(form["brand"])
        if form.get("queue"):
            params["queue"] = "1"
        back = "/brands/photos"
        if params:
            back += "?" + urlencode(params)
        return RedirectResponse(url=back, status_code=303)

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
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return FileResponse(path, media_type="image/jpeg")

    return app


def build() -> FastAPI:
    """uvicorn factory target (see Makefile run)."""
    return create_app()
