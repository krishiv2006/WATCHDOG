"""Data-quality checks against the active baseline schema."""

from __future__ import annotations

from collections import Counter
from datetime import timedelta

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models import BaselineVersion, PredictionLog
from app.models import utcnow


def schema_warnings(
    features: dict,
    baseline: BaselineVersion | None,
) -> list[str]:
    """Unknown / missing features relative to the active baseline profile."""
    if not baseline or not baseline.feature_profiles:
        return []
    expected = {p["name"] for p in baseline.feature_profiles}
    incoming = set(features.keys())
    warnings = []
    missing = expected - incoming
    extra = incoming - expected
    if missing:
        warnings.append(f"missing features: {sorted(missing)[:8]}")
    if extra:
        warnings.append(f"unknown features: {sorted(extra)[:8]}")
    return warnings


def volume_anomaly(
    db: Session,
    model_id: str,
    window_hours: int = 24,
    lookback_windows: int = 7,
    factor: float = 3.0,
) -> dict | None:
    """Compare last-window prediction count to the rolling mean of prior windows.

    Returns a dict with current/expected counts when the drop or spike exceeds
    ``factor`` times (or is below 1/factor of) the rolling average.
    """
    now = utcnow()
    window = timedelta(hours=window_hours)
    current = (
        db.query(func.count(PredictionLog.id))
        .filter(
            PredictionLog.model_id == model_id,
            PredictionLog.timestamp >= now - window,
        )
        .scalar()
        or 0
    )
    prior_counts = []
    for i in range(1, lookback_windows + 1):
        start = now - window * (i + 1)
        end = now - window * i
        c = (
            db.query(func.count(PredictionLog.id))
            .filter(
                PredictionLog.model_id == model_id,
                PredictionLog.timestamp >= start,
                PredictionLog.timestamp < end,
            )
            .scalar()
            or 0
        )
        prior_counts.append(c)
    if not prior_counts or sum(prior_counts) == 0:
        return None
    avg = sum(prior_counts) / len(prior_counts)
    if avg < 10:
        return None  # not enough history to call this an anomaly
    if current > avg * factor or current < avg / factor:
        return {
            "current": int(current),
            "expected": round(avg, 1),
            "ratio": round(current / avg, 3) if avg else None,
        }
    return None


def duplicate_prediction_ids(db: Session, model_id: str) -> int:
    rows = (
        db.query(PredictionLog.prediction_id, func.count(PredictionLog.id))
        .filter(
            PredictionLog.model_id == model_id,
            PredictionLog.prediction_id.isnot(None),
        )
        .group_by(PredictionLog.prediction_id)
        .having(func.count(PredictionLog.id) > 1)
        .all()
    )
    return sum(cnt - 1 for _, cnt in rows)


def feature_null_spikes(
    predictions: list[PredictionLog],
    baseline: BaselineVersion,
    spike_delta: float = 0.2,
) -> list[dict]:
    if not predictions or not baseline.feature_profiles:
        return []
    spikes = []
    for profile in baseline.feature_profiles:
        name = profile["name"]
        base_null = float(profile.get("null_rate") or 0)
        n = 0
        nulls = 0
        for p in predictions:
            fv = p.feature_values or {}
            n += 1
            if name not in fv or fv[name] is None:
                nulls += 1
        live_null = nulls / n if n else 0.0
        if live_null - base_null >= spike_delta:
            spikes.append({
                "feature": name,
                "baselineNullRate": round(base_null, 4),
                "liveNullRate": round(live_null, 4),
            })
    return spikes


def unseen_categories(live_values: list, frequencies: dict) -> list[str]:
    known = set(frequencies.keys()) | {"__other__"}
    counts = Counter(str(v) for v in live_values)
    return sorted(k for k in counts if k not in known)
