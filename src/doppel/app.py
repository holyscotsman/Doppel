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

from doppel.config import Config, load_config
from doppel.db import connect
from doppel.drive import (
    CredentialsRequired,
    FetchError,
    GoogleDriveClient,
    ImageFetcher,
    get_credentials,
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


def _build_real_fetcher(config: Config) -> ImageFetcher:
    from google.auth.transport.requests import AuthorizedSession

    from doppel.drive import DriveImageFetcher

    creds = get_credentials(allow_interactive=False)
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
    vlm_factory: Callable[[Config], VlmClient] | None = None,
) -> FastAPI:
    """Build the app. Tests inject a Config and fake factories."""
    config = config or load_config()
    fetcher_factory = fetcher_factory or _build_real_fetcher
    embedder_factory = embedder_factory or (lambda cfg: ClipEmbedder(cfg.clip_model))
    vlm_factory = vlm_factory or (
        lambda cfg: OllamaClient(cfg.ollama.host, cfg.ollama.model)
    )

    app = FastAPI(title="doppel")
    app.state.config = config
    app.state.runner = JobRunner()
    app.state.fetcher = None  # built lazily: needs OAuth credentials
    app.state.embedder = None  # built lazily: loads the CLIP model
    app.state.vlm = None  # built lazily: needs the Ollama server

    templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))
    app.mount(
        "/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static"
    )

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
                    creds = get_credentials(allow_interactive=False)
                    client = GoogleDriveClient(creds)
                    job = lambda: run_sync(conn, client, config.cache_dir)  # noqa: E731
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
        if stage == "sync" and not (
            Path("token.json").exists() or Path("credentials.json").exists()
        ):
            error = (
                "credentials.json not found — create an OAuth client "
                "(Desktop app) and place it at the repo root."
            )
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
            WHERE d.action = 'trash' ORDER BY p.name
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
