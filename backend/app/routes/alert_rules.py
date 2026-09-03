"""Alert rule management endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import AlertRule
from app.schemas import AlertRuleCreate, AlertRuleUpdate, AlertRuleOut

router = APIRouter(prefix="/api", tags=["alert-rules"])


@router.get("/alert-rules")
def list_alert_rules(model_id: str | None = None, db: Session = Depends(get_db)):
    """List alert rules, optionally filtered by model_id."""
    query = db.query(AlertRule)
    if model_id:
        query = query.filter(
            (AlertRule.model_id == model_id) | (AlertRule.model_id.is_(None))
        )
    rules = query.all()
    return [AlertRuleOut.model_validate(r) for r in rules]


@router.post("/alert-rules", status_code=201)
def create_alert_rule(payload: AlertRuleCreate, db: Session = Depends(get_db)):
    """Create a new alert rule."""
    rule = AlertRule(**payload.model_dump())
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return AlertRuleOut.model_validate(rule)


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
    return AlertRuleOut.model_validate(rule)
