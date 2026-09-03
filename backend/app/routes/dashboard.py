"""Dashboard endpoint — aggregates data across models for the overview page."""

from datetime import datetime, timedelta
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.models import (
    Model as ModelDB, PredictionLog, DriftScore,
    PerformanceSnapshot, Alert as AlertDB,
)

router = APIRouter(prefix="/api", tags=["dashboard"])


@router.get("/dashboard/summary")
def get_dashboard_summary(db: Session = Depends(get_db)):
    """Aggregated dashboard summary matching the frontend contract."""

    # ── Models ─────────────────────────────────────
    models = db.query(ModelDB).filter(ModelDB.status.in_(["production", "staging", "retraining"])).all()
    prod_count = sum(1 for m in models if m.status == "production")
    staging_count = sum(1 for m in models if m.status in ("staging", "retraining"))
    health_avg = round(sum(m.health_score for m in models) / max(len(models), 1), 1)

    # ── Predictions count ──────────────────────────
    pred_count = db.query(func.count(PredictionLog.id)).scalar() or 0

    # ── Alerts ─────────────────────────────────────
    active_alerts = db.query(AlertDB).filter(AlertDB.status != "resolved").all()
    critical_count = sum(1 for a in active_alerts if a.severity == "critical")
    warning_count = sum(1 for a in active_alerts if a.severity == "warning")

    # ── Health over time (last 30 days, weekly) ────
    health_over_time = []
    now = datetime.utcnow()
    for week in range(5, 0, -1):
        ts = now - timedelta(weeks=week)
        # Get average performance snapshot closest to that date for all models
        snapshots = db.query(PerformanceSnapshot).filter(
            PerformanceSnapshot.timestamp >= ts - timedelta(days=3),
            PerformanceSnapshot.timestamp <= ts + timedelta(days=3),
        ).all()
        if snapshots:
            avg_accuracy = sum(s.accuracy for s in snapshots if s.accuracy) / len(snapshots)
            health_over_time.append({
                "time": f"W-{week}",
                "value": round(avg_accuracy * 100, 1),
            })
        else:
            health_over_time.append({"time": f"W-{week}", "value": health_avg})

    # ── Top feature drift alerts ───────────────────
    top_drifts = db.query(DriftScore).order_by(DriftScore.psi_score.desc()).limit(4).all()
    feature_drift_alerts = []
    for d in top_drifts:
        feature_drift_alerts.append({
            "feature": d.feature_name,
            "score": f"{d.psi_score:.2f}",
            "status": d.status,
        })

    return {
        "healthAvg": health_avg,
        "activeModels": {
            "total": len(models),
            "prod": prod_count,
            "staging": staging_count,
        },
        "predictionsTotal": pred_count,
        "activeAlerts": {
            "total": len(active_alerts),
            "critical": critical_count,
            "warning": warning_count,
        },
        "healthOverTime": health_over_time,
        "featureDriftAlerts": feature_drift_alerts,
    }
