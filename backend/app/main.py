"""Watchdog AI — FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import engine, Base
from app.ws_manager import ws_manager
from app.routes import dashboard, models, alerts, alert_rules, reports

settings = get_settings()

# ── Logging ────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("watchdog")


# ── Lifespan ───────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure schema and runtime state are ready before serving traffic."""
    alembic_cfg = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    try:
        logger.info("Running database migrations...")
        command.upgrade(alembic_cfg, "head")
    except Exception as exc:  # pragma: no cover - startup safeguard
        logger.warning("Alembic migration failed during startup: %s", exc)
        logger.info("Falling back to create_all for legacy/dev databases.")
        Base.metadata.create_all(bind=engine)
    if settings.ENABLE_SCHEDULER:
        from app.services.scheduler import start_scheduler, shutdown_scheduler
        start_scheduler()
    logger.info("Watchdog AI backend ready.")
    yield
    if settings.ENABLE_SCHEDULER:
        from app.services.scheduler import shutdown_scheduler
        shutdown_scheduler()
    logger.info("Shutting down Watchdog AI backend.")


# ── App ────────────────────────────────────────────
app = FastAPI(
    title="Watchdog AI",
    description="ML model monitoring service — detects data drift, prediction drift, and performance degradation.",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ── Register route modules ────────────────────────
app.include_router(dashboard.router)
app.include_router(models.router)
app.include_router(alerts.router)
app.include_router(alert_rules.router)
app.include_router(reports.router)


# ── WebSocket endpoint ────────────────────────────
@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await ws_manager.connect(websocket)
    logger.info(f"WebSocket client connected. Total: {len(ws_manager.active_connections)}")
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(ws_manager.active_connections)}")


# ── Health / readiness ─────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "watchdog-ai", "version": "2.0.0"}


@app.get("/ready")
def readiness():
    from sqlalchemy import text
    from app.database import SessionLocal
    from app.services import scheduler as sched_mod

    db_ok = False
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        db_ok = True
    except Exception as exc:
        logger.warning("Readiness DB check failed: %s", exc)
    sched_ok = (not settings.ENABLE_SCHEDULER) or (sched_mod._scheduler is not None)
    status = "ready" if db_ok and sched_ok else "degraded"
    code = 200 if db_ok else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=code,
        content={"status": status, "database": db_ok, "scheduler": sched_ok},
    )
