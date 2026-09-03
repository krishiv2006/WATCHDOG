"""Watchdog AI — FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
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
    """Create tables on startup (idempotent)."""
    logger.info("Creating database tables (if not exist)...")
    Base.metadata.create_all(bind=engine)
    logger.info("Watchdog AI backend ready.")
    yield
    logger.info("Shutting down Watchdog AI backend.")


# ── App ────────────────────────────────────────────
app = FastAPI(
    title="Watchdog AI",
    description="ML model monitoring service — detects data drift, prediction drift, and performance degradation.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
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
            # Keep connection alive, listen for client messages (pings)
            data = await websocket.receive_text()
            # Echo back or handle keepalive
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(ws_manager.active_connections)}")


# ── Health check ──────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "watchdog-ai"}
