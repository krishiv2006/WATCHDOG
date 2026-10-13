"""Alert evaluation: fingerprint de-dup, persistence, cooldown, auto-resolve.

Reads ``AlertRule`` (per-model overriding global, then config). Broadcasts
``new_alert`` / ``alert_updated`` over the WebSocket manager.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Alert as AlertDB, AlertRule, DriftScore, utcnow
from app.ws_manager import ws_manager

logger = logging.getLogger(__name__)
settings = get_settings()


def _broadcast(payload: dict) -> None:
    """Fire-and-forget broadcast that works from sync scheduler threads."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(ws_manager.broadcast(payload))
    except RuntimeError:
        try:
            asyncio.run(ws_manager.broadcast(payload))
        except Exception:
            logger.debug("WebSocket broadcast skipped (no event loop)", exc_info=True)


def _alert_dict(alert: AlertDB, model_name: str) -> dict:
    return {
        "id": alert.id,
        "modelId": alert.model_id,
        "modelName": model_name,
        "severity": alert.severity,
        "category": alert.category,
        "title": alert.title,
        "description": alert.description,
        "detectedAt": alert.detected_at.isoformat() if alert.detected_at else None,
        "status": alert.status,
        "fingerprint": alert.fingerprint,
        "occurrences": alert.occurrences,
        "firstSeenAt": alert.first_seen_at.isoformat() if alert.first_seen_at else None,
        "lastSeenAt": alert.last_seen_at.isoformat() if alert.last_seen_at else None,
        "autoResolved": alert.auto_resolved,
        "metricValue": alert.metric_value,
    }


def resolve_rule(
    db: Session,
    model_id: str,
    metric: str,
    feature_name: str | None = None,
) -> AlertRule | None:
    q = db.query(AlertRule).filter(AlertRule.metric == metric, AlertRule.enabled == True)  # noqa: E712
    if feature_name:
        specific = q.filter(AlertRule.model_id == model_id, AlertRule.feature_name == feature_name).first()
        if specific:
            return specific
    per_model = q.filter(AlertRule.model_id == model_id, AlertRule.feature_name.is_(None)).first()
    if per_model:
        return per_model
    return q.filter(AlertRule.model_id.is_(None)).first()


def consecutive_flagged(
    db: Session,
    model_id: str,
    feature_name: str,
    n: int,
) -> bool:
    """True if the last ``n`` DriftScore rows for this feature are warning/critical."""
    if n <= 1:
        return True
    rows = (
        db.query(DriftScore)
        .filter(DriftScore.model_id == model_id, DriftScore.feature_name == feature_name)
        .order_by(DriftScore.timestamp.desc())
        .limit(n)
        .all()
    )
    if len(rows) < n:
        return False
    return all(r.status in ("warning", "critical") for r in rows)


def upsert_drift_alert(
    db: Session,
    model_id: str,
    model_name: str,
    feature_name: str,
    status: str,
    psi: float,
    js: float,
    n_live: int,
    reason: str,
    baseline_mean: float | None,
    current_mean: float | None,
) -> AlertDB | None:
    """Create or update a drift alert. Returns the alert or None if suppressed."""
    now = utcnow()
    fingerprint = f"{model_id}:data_drift:{feature_name}"
    rule = resolve_rule(db, model_id, "psi", feature_name)
    consecutive = (
        rule.consecutive_windows
        if rule and rule.consecutive_windows
        else settings.DRIFT_CONSECUTIVE_WINDOWS
    )
    cooldown = rule.cooldown_minutes if rule and rule.cooldown_minutes else 60

    open_alert = (
        db.query(AlertDB)
        .filter(
            AlertDB.fingerprint == fingerprint,
            AlertDB.status != "resolved",
        )
        .first()
    )

    if status not in ("warning", "critical"):
        if open_alert:
            open_alert.status = "resolved"
            open_alert.resolved_at = now
            open_alert.auto_resolved = True
            open_alert.last_seen_at = now
            db.flush()
            _broadcast({"event": "alert_updated", "alert": _alert_dict(open_alert, model_name)})
        return None

    if not consecutive_flagged(db, model_id, feature_name, consecutive):
        logger.debug(
            "Suppressing %s alert for %s:%s — persistence %s not met",
            status, model_id, feature_name, consecutive,
        )
        return None

    description = (
        f"Feature '{feature_name}': PSI={psi:.4f}, JS={js:.4f}, n_live={n_live}. "
        f"Baseline mean={baseline_mean}, current mean={current_mean}. {reason}"
    )
    title = f"Data Drift Detected: '{feature_name}' ({status.upper()})"

    if open_alert:
        # Cooldown: still bump last_seen/occurrences, optionally escalate
        last = open_alert.last_seen_at or open_alert.detected_at
        if last and last.tzinfo is None:
            last = last.replace(tzinfo=now.tzinfo)
        within_cooldown = last is not None and (now - last) < timedelta(minutes=cooldown)
        open_alert.last_seen_at = now
        open_alert.occurrences = (open_alert.occurrences or 1) + 1
        open_alert.metric_value = psi
        open_alert.description = description
        if status == "critical" and open_alert.severity != "critical":
            open_alert.severity = "critical"
            open_alert.title = title
        if not within_cooldown:
            open_alert.title = title
        db.flush()
        _broadcast({"event": "alert_updated", "alert": _alert_dict(open_alert, model_name)})
        return open_alert

    alert = AlertDB(
        model_id=model_id,
        severity=status,
        category="data_drift",
        title=title,
        description=description,
        detected_at=now,
        status="unacknowledged",
        fingerprint=fingerprint,
        first_seen_at=now,
        last_seen_at=now,
        occurrences=1,
        auto_resolved=False,
        alert_rule_id=rule.id if rule else None,
        metric_value=psi,
    )
    db.add(alert)
    db.flush()
    _broadcast({"event": "new_alert", "alert": _alert_dict(alert, model_name)})
    return alert


def upsert_quality_alert(
    db: Session,
    model_id: str,
    model_name: str,
    kind: str,
    title: str,
    description: str,
    metric_value: float | None = None,
) -> AlertDB:
    now = utcnow()
    fingerprint = f"{model_id}:data_quality:{kind}"
    open_alert = (
        db.query(AlertDB)
        .filter(AlertDB.fingerprint == fingerprint, AlertDB.status != "resolved")
        .first()
    )
    if open_alert:
        open_alert.last_seen_at = now
        open_alert.occurrences = (open_alert.occurrences or 1) + 1
        open_alert.description = description
        open_alert.metric_value = metric_value
        db.flush()
        _broadcast({"event": "alert_updated", "alert": _alert_dict(open_alert, model_name)})
        return open_alert
    alert = AlertDB(
        model_id=model_id,
        severity="warning",
        category="data_quality",
        title=title,
        description=description,
        detected_at=now,
        status="unacknowledged",
        fingerprint=fingerprint,
        first_seen_at=now,
        last_seen_at=now,
        occurrences=1,
        metric_value=metric_value,
    )
    db.add(alert)
    db.flush()
    _broadcast({"event": "new_alert", "alert": _alert_dict(alert, model_name)})
    return alert
