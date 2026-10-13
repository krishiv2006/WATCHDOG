"""Dashboard endpoint — aggregates data across models for the overview page."""

from datetime import timedelta
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.config import get_settings
from app.models import (
    Model as ModelDB, PredictionLog, DriftScore,
    Alert as AlertDB, utcnow,
)

router = APIRouter(prefix="/api", tags=["dashboard"])


@router.get("/dashboard/summary")
def get_dashboard_summary(window: str = "30d", db: Session = Depends(get_db)):
    """Aggregated dashboard summary matching the frontend contract."""
    from app.routes.models import parse_window
    from app.services.drift import compute_health_score
    from app.config import get_settings

    days = parse_window(window)
    now = utcnow()
    cutoff = now - timedelta(days=days)
    settings = get_settings()

    # ── Models ─────────────────────────────────────
    models = db.query(ModelDB).filter(
        ModelDB.status.in_(["production", "staging", "retraining"]),
        ModelDB.retired_at.is_(None),
    ).all()
    prod_count = sum(1 for m in models if m.status == "production")
    staging_count = sum(1 for m in models if m.status in ("staging", "retraining"))
    health_avg = round(sum((m.health_score or 0) for m in models) / max(len(models), 1), 1) if models else None

    # ── Predictions count ──────────────────────────
    pred_count = (
        db.query(func.count(PredictionLog.id))
        .filter(PredictionLog.timestamp >= cutoff)
        .scalar()
        or 0
    )
    pred_prev = (
        db.query(func.count(PredictionLog.id))
        .filter(
            PredictionLog.timestamp >= cutoff - timedelta(days=days),
            PredictionLog.timestamp < cutoff,
        )
        .scalar()
        or 0
    )
    pred_trend = None
    if pred_prev > 0:
        pred_trend = round((pred_count - pred_prev) / pred_prev * 100, 1)

    # ── Alerts ─────────────────────────────────────
    active_alerts = db.query(AlertDB).filter(AlertDB.status != "resolved").all()
    critical_count = sum(1 for a in active_alerts if a.severity == "critical")
    warning_count = sum(1 for a in active_alerts if a.severity == "warning")

    # ── Health over time: reconstruct from drift PSI per day, not accuracy ──
    health_over_time = []
    buckets = min(days, 12)
    bucket_days = max(1, days // buckets)
    for i in range(buckets, 0, -1):
        start = now - timedelta(days=i * bucket_days)
        end = now - timedelta(days=(i - 1) * bucket_days)
        scores = (
            db.query(DriftScore.psi_score)
            .filter(DriftScore.timestamp >= start, DriftScore.timestamp < end)
            .all()
        )
        psis = [s[0] for s in scores if s[0] is not None]
        if psis:
            hs = compute_health_score(
                avg_psi=sum(psis) / len(psis),
                open_critical_alerts=0,
                psi_warning=settings.PSI_THRESHOLD_WARNING,
                psi_critical=settings.PSI_THRESHOLD_CRITICAL,
                performance_ratio=None,
            )
            health_over_time.append({"time": start.strftime("%b %d"), "value": hs.score})
        else:
            health_over_time.append({"time": start.strftime("%b %d"), "value": None})

    # ── Top feature drift (latest per feature, real rows only) ──
    top_drifts = (
        db.query(DriftScore)
        .filter(DriftScore.timestamp >= cutoff, DriftScore.psi_score.isnot(None))
        .order_by(DriftScore.psi_score.desc())
        .limit(8)
        .all()
    )
    seen = set()
    feature_drift_alerts = []
    for d in top_drifts:
        key = (d.model_id, d.feature_name)
        if key in seen:
            continue
        seen.add(key)
        feature_drift_alerts.append({
            "feature": d.feature_name,
            "score": f"{d.psi_score:.2f}" if d.psi_score is not None else "n/a",
            "status": d.status,
        })
        if len(feature_drift_alerts) >= 4:
            break

    health_badge = "Unknown"
    if health_avg is not None:
        if health_avg >= 80:
            health_badge = "Healthy"
        elif health_avg >= 50:
            health_badge = "Degraded"
        else:
            health_badge = "Critical"

    return {
        "healthAvg": health_avg,
        "healthBadge": health_badge,
        "healthTrend": None,
        "activeModels": {
            "total": len(models),
            "prod": prod_count,
            "staging": staging_count,
        },
        "predictionsTotal": pred_count,
        "predictionsTrend": pred_trend,
        "activeAlerts": {
            "total": len(active_alerts),
            "critical": critical_count,
            "warning": warning_count,
        },
        "healthOverTime": health_over_time,
        "featureDriftAlerts": feature_drift_alerts,
    }


@router.get("/settings")
def public_settings():
    """Non-secret runtime config the UI can display honestly."""
    s = get_settings()
    return {
        "ingestAuthRequired": bool(s.INGEST_API_KEY),
        "ingestHeader": "X-API-Key",
        "evalWindowHours": s.EVAL_WINDOW_HOURS,
        "driftCheckIntervalMinutes": s.DRIFT_CHECK_INTERVAL_MINUTES,
        "minSampleSize": s.MIN_SAMPLE_SIZE,
        "psiWarning": s.PSI_THRESHOLD_WARNING,
        "psiCritical": s.PSI_THRESHOLD_CRITICAL,
        "schedulerEnabled": s.ENABLE_SCHEDULER,
    }
