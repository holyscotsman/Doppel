"""FastAPI app: routes, templates, job wiring. Server-rendered UI (Jinja2 +
htmx); binds to 127.0.0.1 only (see Makefile run target)."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from doppel.config import Config, load_config
from doppel.db import connect
from doppel.drive import GoogleDriveClient, ImageFetcher, get_credentials
from doppel.embed import ClipEmbedder, Embedder
from doppel.jobs import JobRunner, run_sync
from doppel.stages.exact import run_exact
from doppel.stages.near import run_near
from doppel.stages.similar import run_similar

PACKAGE_DIR = Path(__file__).parent

# stages the UI can launch, in pipeline order; extended phase by phase
UI_STAGES = ["sync", "exact", "near", "similar"]

PAGE_SIZE = 20


def _build_real_fetcher(config: Config) -> ImageFetcher:
    from google.auth.transport.requests import AuthorizedSession

    from doppel.drive import DriveImageFetcher

    creds = get_credentials()
    return DriveImageFetcher(
        db_path=config.db_path,
        client=GoogleDriveClient(creds),
        session=AuthorizedSession(creds),
        cache_dir=config.cache_dir,
    )


def create_app(
    config: Config | None = None,
    fetcher_factory: Callable[[Config], ImageFetcher] | None = None,
    embedder_factory: Callable[[Config], Embedder] | None = None,
) -> FastAPI:
    """Build the app. Tests inject a Config and fake fetcher/embedder factories."""
    config = config or load_config()
    fetcher_factory = fetcher_factory or _build_real_fetcher
    embedder_factory = embedder_factory or (lambda cfg: ClipEmbedder(cfg.clip_model))

    app = FastAPI(title="doppel")
    app.state.config = config
    app.state.runner = JobRunner()
    app.state.fetcher = None  # built lazily: needs OAuth credentials
    app.state.embedder = None  # built lazily: loads the CLIP model

    templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))
    app.mount(
        "/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static"
    )

    def get_conn() -> Iterator[sqlite3.Connection]:
        conn = connect(config.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def get_fetcher() -> ImageFetcher:
        if app.state.fetcher is None:
            app.state.fetcher = fetcher_factory(config)
        return app.state.fetcher

    def get_embedder() -> Embedder:
        if app.state.embedder is None:
            app.state.embedder = embedder_factory(config)
        return app.state.embedder

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
            if stage == "sync":
                creds = get_credentials()
                run_sync(conn, GoogleDriveClient(creds))
            elif stage == "exact":
                run_exact(conn)
            elif stage == "near":
                run_near(conn, get_fetcher(), config)
            elif stage == "similar":
                run_similar(conn, get_fetcher(), get_embedder(), config)
        finally:
            conn.close()

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
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "photo_count": photo_count,
                "tier_counts": tier_counts,
                "scans": scan_overview(conn),
            },
        )

    @app.get("/partials/scans", response_class=HTMLResponse)
    def scans_partial(request: Request, conn: sqlite3.Connection = Depends(get_conn)):
        return templates.TemplateResponse(
            request, "_scan_status.html", {"scans": scan_overview(conn), "error": None}
        )

    @app.post("/scans/{stage}", response_class=HTMLResponse)
    def start_stage(
        request: Request, stage: str, conn: sqlite3.Connection = Depends(get_conn)
    ):
        error = None
        if stage not in UI_STAGES:
            raise HTTPException(status_code=404, detail=f"unknown stage {stage!r}")
        if stage == "sync" and not (
            Path("token.json").exists() or Path("credentials.json").exists()
        ):
            error = (
                "credentials.json not found — create an OAuth client "
                "(Desktop app) and place it at the repo root."
            )
        elif not app.state.runner.start(stage, lambda: run_stage_job(stage)):
            error = "a job is already running"
        return templates.TemplateResponse(
            request, "_scan_status.html", {"scans": scan_overview(conn), "error": error}
        )

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
        conn: sqlite3.Connection = Depends(get_conn),
    ):
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM groups WHERE tier = ?", (tier,)
        ).fetchone()["n"]
        groups = conn.execute(
            """
            SELECT g.id, g.tier, g.color_variant, COUNT(m.photo_id) AS members
            FROM groups g JOIN group_members m ON m.group_id = g.id
            WHERE g.tier = ?
            GROUP BY g.id ORDER BY g.id
            LIMIT ? OFFSET ?
            """,
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
        return templates.TemplateResponse(
            request, "group_detail.html", {"group": group, "members": members}
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
        path = fetcher.get(row["drive_id"], config.thumb_size)
        return FileResponse(path, media_type="image/jpeg")

    return app


def build() -> FastAPI:
    """uvicorn factory target (see Makefile run)."""
    return create_app()
