"""Alert rule management endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import AlertRule
from app.schemas import AlertRuleCreate, AlertRuleUpdate, AlertRuleOut

router = APIRouter(prefix="/api", tags=["alert-rules"])


def _rule_to_dict(rule: AlertRule) -> dict:
    """Serialize AlertRule to camelCase dict for frontend."""
    return {
        "id": rule.id,
        "modelId": rule.model_id,
        "metric": rule.metric,
        "thresholdWarning": rule.threshold_warning,
        "thresholdCritical": rule.threshold_critical,
        "enabled": rule.enabled,
        "featureName": rule.feature_name,
        "consecutiveWindows": rule.consecutive_windows,
        "cooldownMinutes": rule.cooldown_minutes,
    }


@router.get("/alert-rules")
def list_alert_rules(model_id: str | None = None, db: Session = Depends(get_db)):
    """List alert rules, optionally filtered by model_id."""
    query = db.query(AlertRule)
    if model_id:
        query = query.filter(
            (AlertRule.model_id == model_id) | (AlertRule.model_id.is_(None))
        )
    rules = query.all()
    return [_rule_to_dict(r) for r in rules]


@router.post("/alert-rules", status_code=201)
def create_alert_rule(payload: AlertRuleCreate, db: Session = Depends(get_db)):
    """Create a new alert rule."""
    rule = AlertRule(
        model_id=payload.model_id,
        metric=payload.metric,
        threshold_warning=payload.threshold_warning,
        threshold_critical=payload.threshold_critical,
        enabled=payload.enabled,
        feature_name=payload.feature_name,
        consecutive_windows=payload.consecutive_windows,
        cooldown_minutes=payload.cooldown_minutes,
    )
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return _rule_to_dict(rule)


@router.patch("/alert-rules/{rule_id}")
def update_alert_rule(rule_id: str, payload: AlertRuleUpdate, db: Session = Depends(get_db)):
    """Update an existing alert rule."""
    rule = db.query(AlertRule).filter(AlertRule.id == rule_id).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Alert rule not found")

    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(rule, key, value)

    db.commit()
    db.refresh(rule)
    return _rule_to_dict(rule)


@router.delete("/alert-rules/{rule_id}", status_code=204)
def delete_alert_rule(rule_id: str, db: Session = Depends(get_db)):
    """Delete an alert rule."""
    rule = db.query(AlertRule).filter(AlertRule.id == rule_id).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    db.delete(rule)
    db.commit()
