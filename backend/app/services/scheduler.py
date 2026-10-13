"""APScheduler job that evaluates every active model on an interval."""

from __future__ import annotations

import logging
import threading

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.config import get_settings
from app.database import SessionLocal
from app.models import Model as ModelDB

logger = logging.getLogger(__name__)
settings = get_settings()

_scheduler: BackgroundScheduler | None = None
_lock = threading.Lock()


def run_all_models() -> None:
    if not _lock.acquire(blocking=False):
        logger.info("Skipping drift job — previous run still holding the lock")
        return
    try:
        db = SessionLocal()
        try:
            from app.services.evaluate import evaluate_model

            models = (
                db.query(ModelDB)
                .filter(ModelDB.retired_at.is_(None), ModelDB.status != "retired")
                .all()
            )
            for model in models:
                try:
                    result = evaluate_model(model, db)
                    logger.info("Evaluated %s: %s", model.id, result.get("status"))
                except Exception:
                    logger.exception("Evaluation failed for model %s", model.id)
                    db.rollback()
        finally:
            db.close()
    finally:
        _lock.release()


def generate_report_job(report_id: str) -> None:
    from app.routes.reports import generate_report_file

    generate_report_file(report_id)


def start_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    sched = BackgroundScheduler(job_defaults={"coalesce": True, "max_instances": 1})
    sched.add_job(
        run_all_models,
        trigger=IntervalTrigger(minutes=settings.DRIFT_CHECK_INTERVAL_MINUTES),
        id="drift_check",
        replace_existing=True,
    )
    sched.start()
    _scheduler = sched
    logger.info(
        "Scheduler started: drift check every %s minutes",
        settings.DRIFT_CHECK_INTERVAL_MINUTES,
    )
    return sched


def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
