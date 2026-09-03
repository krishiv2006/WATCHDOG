"""Alert endpoints — list, filter, paginate, update status."""

import math
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db
from app.models import Alert as AlertDB, Model as ModelDB
from app.schemas import AlertUpdate
from app.ws_manager import ws_manager

router = APIRouter(prefix="/api", tags=["alerts"])


def _format_time_ago(dt: datetime) -> str:
    if not dt:
        return "N/A"
    delta = datetime.utcnow() - dt
    if delta.total_seconds() < 60:
        return f"{int(delta.total_seconds())} secs ago"
    if delta.total_seconds() < 3600:
        return f"{int(delta.total_seconds() / 60)} mins ago"
    if delta.total_seconds() < 86400:
        hours = int(delta.total_seconds() / 3600)
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    return f"{int(delta.days)} days ago"


def _alert_to_dict(alert: AlertDB, db: Session) -> dict:
    model = db.query(ModelDB).filter(ModelDB.id == alert.model_id).first()
    model_name = f"{model.name} {model.version}" if model else "Unknown"
    return {
        "id": alert.id,
        "modelId": alert.model_id,
        "modelName": model_name,
        "severity": alert.severity,
        "category": alert.category,
        "title": alert.title,
        "description": alert.description,
        "detectedAt": _format_time_ago(alert.detected_at),
        "status": alert.status,
    }


@router.get("/alerts")
def list_alerts(
    status: str | None = Query(None),
    severity: str | None = Query(None),
    model_id: str | None = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Paginated alert list with optional filters."""
    query = db.query(AlertDB)

    if status:
        query = query.filter(AlertDB.status == status)
    if severity:
        query = query.filter(AlertDB.severity == severity)
    if model_id:
        query = query.filter(AlertDB.model_id == model_id)

    total = query.count()
    total_pages = max(1, math.ceil(total / per_page))

    alerts = query.order_by(desc(AlertDB.detected_at)).offset(
        (page - 1) * per_page
    ).limit(per_page).all()

    return {
        "items": [_alert_to_dict(a, db) for a in alerts],
        "total": total,
        "page": page,
        "perPage": per_page,
        "totalPages": total_pages,
    }


@router.patch("/alerts/{alert_id}")
async def update_alert(alert_id: str, payload: AlertUpdate, db: Session = Depends(get_db)):
    """Update alert status (acknowledge/investigate/resolve)."""
    alert = db.query(AlertDB).filter(AlertDB.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.status = payload.status
    if payload.status == "resolved":
        alert.resolved_at = datetime.utcnow()

    db.commit()
    db.refresh(alert)

    # Broadcast update over WebSocket
    alert_data = _alert_to_dict(alert, db)
    await ws_manager.broadcast({
        "event": "alert_updated",
        "alert": alert_data,
    })

    return alert_data
