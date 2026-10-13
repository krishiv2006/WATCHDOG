"""
Model management, real ingestion, drift, and performance endpoints.

Phase 2: adds POST /baseline, POST /predictions, POST /ground-truth,
BaselineVersion support, enriched drift output, and proper reevaluation.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.config import get_settings
from app.deps import require_ingest_key
from app.models import (
    Model as ModelDB, ModelBaseline, BaselineVersion, PredictionLog,
    DriftScore, PerformanceSnapshot, Alert as AlertDB, AlertRule,
    utcnow,
)
from app.schemas import (
    ModelCreate, PredictionInput, PredictionBatch,
    GroundTruthBatch,
)
from app.services.drift import (
    compute_psi, compute_psi_from_histograms,
    compute_js_distance, compute_ks_test,
    drift_decision, compute_health_score, psi_to_status,
)
from app.ws_manager import ws_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["models"])
settings = get_settings()

_VALID_WINDOWS = {"7d": 7, "30d": 30, "90d": 90}


def parse_window(window: str) -> int:
    """Parse a window string like '7d', '30d', '90d' → days integer.

    Raises HTTPException(400) for invalid values instead of silently
    using the old ``"90" in window`` string hack.
    """
    key = window.strip().lower()
    if key not in _VALID_WINDOWS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid window '{window}'. Must be one of: {', '.join(_VALID_WINDOWS)}",
        )
    return _VALID_WINDOWS[key]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _format_time_ago(dt: datetime) -> str:
    """Convert datetime to relative time string."""
    if not dt:
        return "N/A"
    now = _now_utc()
    # Handle naive datetimes from the DB
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
    if delta.total_seconds() < 60:
        return f"{int(delta.total_seconds())} secs ago"
    if delta.total_seconds() < 3600:
        return f"{int(delta.total_seconds() / 60)} mins ago"
    if delta.total_seconds() < 86400:
        return f"{int(delta.total_seconds() / 3600)} hours ago"
    return f"{int(delta.days)} days ago"


def _drift_status_label(model_id: str, db: Session) -> str:
    """Get the overall drift status for a model from latest scores."""
    drifts = (
        db.query(DriftScore)
        .filter(DriftScore.model_id == model_id)
        .order_by(DriftScore.timestamp.desc())
        .limit(20)
        .all()
    )
    if not drifts:
        return "Low"
    statuses = [d.status for d in drifts]
    if any(s == "critical" for s in statuses):
        return "High"
    if any(s == "warning" for s in statuses):
        return "Moderate"
    return "Low"


def _get_thresholds(model_id: str, metric: str, db: Session) -> tuple[float, float]:
    """
    Look up thresholds: per-model rule → global rule → config.

    Returns (warning_threshold, critical_threshold).
    """
    rule = (
        db.query(AlertRule)
        .filter(AlertRule.model_id == model_id, AlertRule.metric == metric, AlertRule.enabled == True)
        .first()
    )
    if rule:
        return rule.threshold_warning, rule.threshold_critical

    global_rule = (
        db.query(AlertRule)
        .filter(AlertRule.model_id.is_(None), AlertRule.metric == metric, AlertRule.enabled == True)
        .first()
    )
    if global_rule:
        return global_rule.threshold_warning, global_rule.threshold_critical

    if metric == "js_distance":
        return settings.JS_THRESHOLD_WARNING, settings.JS_THRESHOLD_CRITICAL
    return settings.PSI_THRESHOLD_WARNING, settings.PSI_THRESHOLD_CRITICAL


def _get_active_baseline(model_id: str, db: Session) -> BaselineVersion | None:
    """Return the active BaselineVersion for a model, or None."""
    return (
        db.query(BaselineVersion)
        .filter(BaselineVersion.model_id == model_id, BaselineVersion.is_active == True)
        .first()
    )


# ── Model CRUD ──────────────────────────────────────────────────────────────


@router.get("/models")
def list_models(db: Session = Depends(get_db)):
    """List all models with summary info for the dashboard table."""
    models = db.query(ModelDB).filter(ModelDB.retired_at.is_(None)).all()
    result = []
    for m in models:
        pred_count = (
            db.query(func.count(PredictionLog.id))
            .filter(PredictionLog.model_id == m.id)
            .scalar()
            or 0
        )
        result.append({
            "id": m.id,
            "name": m.name,
            "version": m.version,
            "algorithm": m.algorithm,
            "environment": m.environment,
            "status": m.status,
            "taskType": m.task_type,
            "predictionsCount": pred_count,
            "healthScore": m.health_score,
            "driftStatus": _drift_status_label(m.id, db),
            "lastUpdated": _format_time_ago(m.last_evaluated_at),
        })
    return result


@router.get("/models/{model_id}")
def get_model(model_id: str, db: Session = Depends(get_db)):
    """Get detailed model info."""
    model = db.query(ModelDB).filter(ModelDB.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    pred_count = (
        db.query(func.count(PredictionLog.id))
        .filter(
            PredictionLog.model_id == model_id,
            PredictionLog.timestamp >= _now_utc() - timedelta(days=7),
        )
        .scalar()
        or 0
    )

    baseline = _get_active_baseline(model_id, db)

    return {
        "id": model.id,
        "name": model.name,
        "version": model.version,
        "algorithm": model.algorithm,
        "environment": model.environment,
        "status": model.status,
        "taskType": model.task_type,
        "predictionsCount": pred_count,
        "healthScore": model.health_score,
        "driftStatus": _drift_status_label(model_id, db),
        "lastUpdated": _format_time_ago(model.last_evaluated_at),
        "trainingDataRef": model.training_data_ref,
        "createdAt": model.created_at.isoformat() if model.created_at else None,
        "hasBaseline": baseline is not None,
        "baselineVersionId": baseline.id if baseline else None,
        "baselineSampleSize": baseline.sample_size if baseline else None,
    }


@router.post("/models", status_code=201)
def create_model(payload: ModelCreate, db: Session = Depends(get_db)):
    """Register a new model."""
    model = ModelDB(
        name=payload.name,
        version=payload.version,
        algorithm=payload.algorithm,
        environment=payload.environment,
        status=payload.status,
        task_type=payload.task_type,
        prediction_column=payload.prediction_column,
        label_delay_hint=payload.label_delay_hint,
        training_data_ref=payload.training_data_ref,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return {"id": model.id, "name": model.name, "status": "created"}


# ── Baseline upload (Section C.1) ───────────────────────────────────────────


def _build_feature_profile(col_name: str, values: np.ndarray, n_buckets: int = 10) -> dict:
    """Build a compact feature profile for storage in BaselineVersion."""
    clean = values[np.isfinite(values)] if values.dtype.kind == 'f' else values
    null_rate = 1.0 - len(clean) / max(len(values), 1)

    # Check if categorical (low cardinality integer or few unique values)
    n_unique = len(np.unique(clean)) if len(clean) > 0 else 0

    if len(clean) == 0:
        return {
            "name": col_name,
            "feature_type": "numeric",
            "count": len(values),
            "null_rate": null_rate,
            "constant": True,
        }

    profile: dict = {
        "name": col_name,
        "feature_type": "numeric",
        "count": int(len(clean)),
        "null_rate": round(null_rate, 6),
    }

    if np.ptp(clean) == 0:
        profile["constant"] = True
        profile["min"] = float(clean[0])
        profile["max"] = float(clean[0])
        profile["mean"] = float(clean[0])
        profile["std"] = 0.0
        return profile

    profile["min"] = float(np.min(clean))
    profile["max"] = float(np.max(clean))
    profile["mean"] = float(np.mean(clean))
    profile["std"] = float(np.std(clean))

    # Percentiles
    pcts = np.percentile(clean, [25, 50, 75])
    profile["percentiles"] = {"25": float(pcts[0]), "50": float(pcts[1]), "75": float(pcts[2])}

    # Quantile edges for PSI (these are the ones actually used for scoring)
    quantiles = np.linspace(0, 100, n_buckets + 1)[1:-1]
    interior_edges = np.unique(np.percentile(clean, quantiles))
    profile["quantile_edges"] = [float(e) for e in interior_edges]
    n_actual = len(interior_edges) + 1
    q_buckets = np.digitize(clean, interior_edges)
    profile["quantile_counts"] = [int(x) for x in np.bincount(q_buckets, minlength=n_actual)[:n_actual]]

    # Fine histogram for JS and UI plots
    hist_counts, hist_edges = np.histogram(clean, bins=min(50, max(10, n_unique)))
    profile["histogram_counts"] = [int(x) for x in hist_counts]
    profile["histogram_edges"] = [float(x) for x in hist_edges]

    return profile


def _build_categorical_profile(col_name: str, series: pd.Series, top_k: int = 50) -> dict:
    """Compact category-frequency sketch for PSI on discrete columns."""
    n = len(series)
    null_rate = float(series.isna().mean()) if n else 0.0
    vals = series.dropna().astype(str)
    counts = vals.value_counts()
    top = counts.head(top_k)
    other = int(counts.iloc[top_k:].sum()) if len(counts) > top_k else 0
    return {
        "name": col_name,
        "feature_type": "categorical",
        "count": int(len(vals)),
        "null_rate": round(null_rate, 6),
        "cardinality": int(counts.size),
        "category_frequencies": {str(k): int(v) for k, v in top.items()},
        "top_k": [str(x) for x in top.index.tolist()],
        "other_count": other,
        "constant": bool(counts.size <= 1),
    }


@router.post("/models/{model_id}/baseline")
async def upload_baseline(
    model_id: str,
    file: UploadFile = File(...),
    n_buckets: int = Form(10),
    db: Session = Depends(get_db),
    _: None = Depends(require_ingest_key),
):
    """
    Upload a training/validation dataset to create a new BaselineVersion.

    Creates an immutable baseline profile with per-feature statistics,
    quantile edges for PSI, and histograms for JS/UI.
    """
    model = db.query(ModelDB).filter(ModelDB.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {settings.MAX_UPLOAD_BYTES} bytes)")

    file_hash = hashlib.sha256(content).hexdigest()

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {str(e)}")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV is empty")

    # Identify feature columns (exclude obvious non-features)
    exclude_lower = {
        "id", "predicted_value", "actual_value", "target", "label",
        "prediction", "predict", "score", "probability", "y_pred",
        "y_true", "y", "index", "timestamp", "date",
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "bool", "category", "string"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c.lower() not in exclude_lower]
    cat_feature_cols = [c for c in cat_cols if c.lower() not in exclude_lower]

    if not feature_cols and not cat_feature_cols:
        raise HTTPException(status_code=400, detail="No feature columns found in CSV")
    if len(feature_cols) + len(cat_feature_cols) > settings.MAX_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many feature columns ({len(feature_cols) + len(cat_feature_cols)}; max {settings.MAX_FEATURES})",
        )

    warnings_list = []

    # Build profiles
    profiles = []
    for col in feature_cols:
        vals = df[col].values.astype(float)
        profile = _build_feature_profile(col, vals, n_buckets)

        # Warn about potential issues
        if profile.get("constant"):
            warnings_list.append(f"Column '{col}' is constant — will be excluded from drift scoring")
        elif profile.get("null_rate", 0) > 0.3:
            warnings_list.append(f"Column '{col}' has {profile['null_rate']*100:.0f}% null values")

        profiles.append(profile)

    for col in cat_feature_cols:
        profile = _build_categorical_profile(col, df[col])
        if profile.get("constant"):
            warnings_list.append(f"Column '{col}' is constant — will be excluded from drift scoring")
        elif profile.get("cardinality", 0) > 200:
            warnings_list.append(
                f"Column '{col}' has high cardinality ({profile['cardinality']}); using top-50 + other bucket"
            )
        profiles.append(profile)

    # Check for prediction/actual columns for baseline performance
    pred_col = next(
        (c for c in df.columns if c.lower() in ("predicted_value", "prediction", "predict", "y_pred")),
        None,
    )
    act_col = next(
        (c for c in df.columns if c.lower() in ("actual_value", "actual", "target", "label", "y_true", "y")),
        None,
    )

    perf_metrics = None
    pred_dist = None
    if pred_col and act_col:
        try:
            preds = pd.to_numeric(df[pred_col], errors="coerce")
            actuals = pd.to_numeric(df[act_col], errors="coerce")
            valid = preds.notna() & actuals.notna()
            n_valid = int(valid.sum())
            if n_valid > 0:
                task = model.task_type or "binary_classification"
                if task in ("binary_classification", "multiclass"):
                    acc = float((preds[valid].round() == actuals[valid].round()).mean())
                    perf_metrics = {"accuracy": round(acc, 4), "sample_size": n_valid}
                else:
                    # Regression
                    residuals = actuals[valid] - preds[valid]
                    mae = float(residuals.abs().mean())
                    rmse = float(np.sqrt((residuals**2).mean()))
                    ss_res = float((residuals**2).sum())
                    ss_tot = float(((actuals[valid] - actuals[valid].mean())**2).sum())
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                    perf_metrics = {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4), "sample_size": n_valid}
        except Exception:
            warnings_list.append("Could not compute baseline performance from prediction/actual columns")

    if pred_col:
        try:
            pred_vals = pd.to_numeric(df[pred_col], errors="coerce").dropna().values
            hist_c, hist_e = np.histogram(pred_vals, bins=20)
            pred_dist = {
                "histogram_counts": [int(x) for x in hist_c],
                "histogram_edges": [float(x) for x in hist_e],
            }
        except Exception:
            pass

    # Deactivate previous baseline versions
    db.query(BaselineVersion).filter(
        BaselineVersion.model_id == model_id,
        BaselineVersion.is_active == True,
    ).update({"is_active": False})

    # Get next version number
    max_ver = (
        db.query(func.max(BaselineVersion.version_number))
        .filter(BaselineVersion.model_id == model_id)
        .scalar()
    ) or 0

    baseline = BaselineVersion(
        model_id=model_id,
        version_number=max_ver + 1,
        is_active=True,
        feature_profiles=profiles,
        sample_size=len(df),
        source_file_name=file.filename,
        source_file_hash=file_hash,
        performance_metrics=perf_metrics,
        prediction_distribution=pred_dist,
        n_psi_buckets=n_buckets,
    )
    db.add(baseline)

    # Also store legacy ModelBaseline rows for backwards compatibility
    for profile in profiles:
        if profile.get("constant"):
            continue
        legacy = ModelBaseline(
            model_id=model_id,
            feature_name=profile["name"],
            mean=profile.get("mean"),
            std=profile.get("std"),
            histogram_counts=profile.get("histogram_counts"),
            histogram_edges=profile.get("histogram_edges"),
        )
        db.add(legacy)

    model.training_data_ref = file.filename
    model.last_evaluated_at = _now_utc()
    db.commit()
    db.refresh(baseline)

    return {
        "status": "success",
        "baselineVersionId": baseline.id,
        "versionNumber": baseline.version_number,
        "featureCount": len(profiles),
        "sampleSize": len(df),
        "performanceMetrics": perf_metrics,
        "warnings": warnings_list,
    }


@router.get("/models/{model_id}/baselines")
def list_baselines(model_id: str, db: Session = Depends(get_db)):
    """List all baseline versions for a model."""
    model = db.query(ModelDB).filter(ModelDB.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    versions = (
        db.query(BaselineVersion)
        .filter(BaselineVersion.model_id == model_id)
        .order_by(BaselineVersion.version_number.desc())
        .all()
    )

    return [
        {
            "id": v.id,
            "versionNumber": v.version_number,
            "isActive": v.is_active,
            "sampleSize": v.sample_size,
            "featureCount": len(v.feature_profiles) if v.feature_profiles else 0,
            "sourceFileName": v.source_file_name,
            "nPsiBuckets": v.n_psi_buckets,
            "performanceMetrics": v.performance_metrics,
            "createdAt": v.created_at.isoformat() if v.created_at else None,
        }
        for v in versions
    ]


# ── Prediction ingestion (Section C.2) ──────────────────────────────────────


@router.post("/models/{model_id}/predictions")
def ingest_predictions(
    model_id: str,
    payload: PredictionBatch,
    db: Session = Depends(get_db),
    _: None = Depends(require_ingest_key),
):
    """
    Ingest single or batch predictions (up to MAX_BATCH_PREDICTIONS).

    Idempotent on prediction_id — duplicates are counted but not re-inserted.
    """
    model = db.query(ModelDB).filter(ModelDB.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if len(payload.predictions) > settings.MAX_BATCH_PREDICTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Batch too large: {len(payload.predictions)} predictions (max {settings.MAX_BATCH_PREDICTIONS})",
        )

    now = _now_utc()
    ingested = 0
    duplicates = 0
    warnings_list = []

    # Collect existing prediction_ids for dedup
    pred_ids_in_batch = [p.prediction_id for p in payload.predictions if p.prediction_id]
    existing_ids = set()
    if pred_ids_in_batch:
        existing = (
            db.query(PredictionLog.prediction_id)
            .filter(
                PredictionLog.model_id == model_id,
                PredictionLog.prediction_id.in_(pred_ids_in_batch),
            )
            .all()
        )
        existing_ids = {row[0] for row in existing}

    from app.services.quality import schema_warnings
    from app.models import generate_uuid

    baseline = _get_active_baseline(model_id, db)
    records = []
    for p in payload.predictions:
        # Idempotency check
        if p.prediction_id and p.prediction_id in existing_ids:
            duplicates += 1
            continue
        for w in schema_warnings(p.features or {}, baseline):
            if w not in warnings_list:
                warnings_list.append(w)

        ts = now
        if p.timestamp:
            try:
                ts = datetime.fromisoformat(p.timestamp)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except ValueError:
                warnings_list.append(f"Invalid timestamp for prediction_id={p.prediction_id}; using now()")
                ts = now

        record = PredictionLog(
            model_id=model_id,
            prediction_id=p.prediction_id or generate_uuid(),
            model_version=p.model_version,
            timestamp=ts,
            feature_values=p.features,
            predicted_value=p.prediction,
            score=p.score,
            latency_ms=p.latency_ms,
        )
        records.append(record)
        ingested += 1

    if records:
        db.bulk_save_objects(records)
        db.commit()

    return {
        "ingested": ingested,
        "duplicates": duplicates,
        "warnings": warnings_list,
    }


# ── Ground truth ingestion (Section C.3) ────────────────────────────────────


@router.post("/models/{model_id}/ground-truth")
def ingest_ground_truth(
    model_id: str,
    payload: GroundTruthBatch,
    db: Session = Depends(get_db),
    _: None = Depends(require_ingest_key),
):
    """
    Submit ground-truth labels for previously logged predictions.

    Joins on prediction_id. Reports how many matched and how many were orphaned.
    """
    model = db.query(ModelDB).filter(ModelDB.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    matched = 0
    orphaned = 0
    warnings_list = []

    for label in payload.labels:
        pred = (
            db.query(PredictionLog)
            .filter(
                PredictionLog.model_id == model_id,
                PredictionLog.prediction_id == label.prediction_id,
            )
            .first()
        )
        if not pred:
            orphaned += 1
            continue

        pred.actual_value = label.actual_value
        if label.labeled_at:
            try:
                pred.labeled_at = datetime.fromisoformat(label.labeled_at)
            except ValueError:
                pred.labeled_at = _now_utc()
        else:
            pred.labeled_at = _now_utc()
        matched += 1

    if matched > 0:
        db.commit()

    return {
        "matched": matched,
        "orphaned": orphaned,
        "warnings": warnings_list,
    }


# ── Ingest dataset (legacy convenience endpoint) ────────────────────────────


@router.post("/models/ingest-dataset")
async def ingest_dataset(
    name: str = Form("New Live Model"),
    version: str = Form("v1.0"),
    algorithm: str = Form("XGBoost"),
    environment: str = Form("production"),
    provided_model_id: str | None = Form(None, alias="model_id"),
    file: UploadFile = File(None),
    db: Session = Depends(get_db),
    _: None = Depends(require_ingest_key),
):
    """Convenience: create (or reuse) a model and upload the CSV as a training baseline.

    Does **not** split the file into fake live data. Send production traffic
    separately via POST /models/{id}/predictions.
    """
    if provided_model_id:
        model = db.query(ModelDB).filter(ModelDB.id == provided_model_id).first()
        if not model:
            model = ModelDB(
                id=provided_model_id, name=name, version=version,
                algorithm=algorithm, environment=environment,
            )
            db.add(model)
    else:
        model = ModelDB(name=name, version=version, algorithm=algorithm, environment=environment)
        db.add(model)
    db.commit()
    db.refresh(model)

    if not file or not file.filename:
        return {
            "status": "success",
            "message": f"Model '{model.name}' registered without a baseline.",
            "model": {
                "id": model.id,
                "name": model.name,
                "version": model.version,
                "algorithm": model.algorithm,
                "environment": model.environment,
                "healthScore": model.health_score,
                "status": model.status,
                "driftStatus": "Low",
            },
            "drifts": [],
        }

    result = await upload_baseline(model_id=model.id, file=file, n_buckets=10, db=db)
    return {
        "status": "success",
        "message": (
            f"Registered '{model.name}' and stored baseline version "
            f"{result.get('versionNumber')} from '{file.filename}'. "
            "Send live predictions to POST /api/models/{id}/predictions."
        ),
        "model": {
            "id": model.id,
            "name": model.name,
            "version": model.version,
            "algorithm": model.algorithm,
            "environment": model.environment,
            "healthScore": model.health_score,
            "status": model.status,
            "driftStatus": "Low",
            "hasBaseline": True,
            "baselineVersionId": result.get("baselineVersionId"),
        },
        "baseline": result,
        "drifts": [],
    }


# ── Drift endpoints ─────────────────────────────────────────────────────────


@router.get("/models/{model_id}/drift")
def get_model_drift(model_id: str, window: str = "30d", db: Session = Depends(get_db)):
    """Latest drift score per feature within the requested window."""
    model = db.query(ModelDB).filter(ModelDB.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    days = parse_window(window)
    cutoff = _now_utc() - timedelta(days=days)

    latest_ts_sq = (
        db.query(
            DriftScore.feature_name,
            func.max(DriftScore.timestamp).label("max_ts"),
        )
        .filter(DriftScore.model_id == model_id, DriftScore.timestamp >= cutoff)
        .group_by(DriftScore.feature_name)
        .subquery()
    )

    drifts = (
        db.query(DriftScore)
        .join(
            latest_ts_sq,
            (DriftScore.feature_name == latest_ts_sq.c.feature_name)
            & (DriftScore.timestamp == latest_ts_sq.c.max_ts),
        )
        .filter(DriftScore.model_id == model_id)
        .order_by(DriftScore.psi_score.desc())
        .all()
    )

    return [
        {
            "featureName": d.feature_name,
            "baselineMean": d.baseline_mean,
            "currentMean": d.current_mean,
            "driftScore": d.psi_score,
            "jsDistance": d.js_distance,
            "ksStatistic": d.ks_statistic,
            "ksPvalue": d.ks_pvalue,
            "status": d.status,
            "confidence": d.confidence,
            "reason": d.reason,
            "nLive": d.n_live,
            "nBaseline": d.n_baseline,
            "nullRateBaseline": d.null_rate_baseline,
            "nullRateLive": d.null_rate_live,
            "outOfRangeRate": d.out_of_range_rate,
            "timestamp": d.timestamp.isoformat() if d.timestamp else None,
        }
        for d in drifts
    ]


@router.get("/models/{model_id}/drift/history")
def get_drift_history(
    model_id: str,
    feature: str = Query(..., description="Feature name to get history for"),
    window: str = Query("30d", description="Time window: 7d, 30d, or 90d"),
    db: Session = Depends(get_db),
):
    """Time-series drift history for a single feature."""
    model = db.query(ModelDB).filter(ModelDB.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    days = parse_window(window)
    cutoff = _now_utc() - timedelta(days=days)

    drifts = (
        db.query(DriftScore)
        .filter(
            DriftScore.model_id == model_id,
            DriftScore.feature_name == feature,
            DriftScore.timestamp >= cutoff,
        )
        .order_by(DriftScore.timestamp)
        .all()
    )

    return [
        {
            "timestamp": d.timestamp.isoformat(),
            "psi": d.psi_score,
            "jsDistance": d.js_distance,
            "status": d.status,
            "baselineMean": d.baseline_mean,
            "currentMean": d.current_mean,
            "nLive": d.n_live,
        }
        for d in drifts
    ]


# ── Performance endpoint ────────────────────────────────────────────────────


@router.get("/models/{model_id}/performance")
def get_model_performance(model_id: str, window: str = "30d", db: Session = Depends(get_db)):
    """Performance metrics + time series for a model."""
    model = db.query(ModelDB).filter(ModelDB.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    days = parse_window(window)
    cutoff = _now_utc() - timedelta(days=days)

    legacy_perf = (
        db.query(ModelBaseline)
        .filter(
            ModelBaseline.model_id == model_id,
            ModelBaseline.feature_name == "__performance__",
        )
        .first()
    )
    active_bv = _get_active_baseline(model_id, db)
    bv_perf = (active_bv.performance_metrics or {}) if active_bv else {}

    snapshots = (
        db.query(PerformanceSnapshot)
        .filter(
            PerformanceSnapshot.model_id == model_id,
            PerformanceSnapshot.timestamp >= cutoff,
        )
        .order_by(PerformanceSnapshot.timestamp)
        .all()
    )

    latest = snapshots[-1] if snapshots else None

    def _metric(label: str, current_val, baseline_val, as_percent: bool = True) -> dict:
        if current_val is None and baseline_val is None:
            return {
                "label": label,
                "value": None,
                "baseline": None,
                "delta": None,
                "reason": "awaiting_labels",
            }
        scale = 100.0 if as_percent else 1.0
        curr = round(current_val * scale, 4 if not as_percent else 1) if current_val is not None else None
        base = round(baseline_val * scale, 4 if not as_percent else 1) if baseline_val is not None else None
        delta = None
        if curr is not None and base is not None:
            delta = round((current_val - baseline_val) * scale, 4 if not as_percent else 1)
        return {"label": label, "value": curr, "baseline": base, "delta": delta}

    base_acc = bv_perf.get("accuracy") if bv_perf.get("accuracy") is not None else (legacy_perf.accuracy if legacy_perf else None)
    base_prec = bv_perf.get("precision") if bv_perf.get("precision") is not None else (legacy_perf.precision_score if legacy_perf else None)
    base_rec = bv_perf.get("recall") if bv_perf.get("recall") is not None else (legacy_perf.recall if legacy_perf else None)
    base_f1 = bv_perf.get("f1") if bv_perf.get("f1") is not None else (legacy_perf.f1 if legacy_perf else None)

    curr_acc = latest.accuracy if latest else None
    curr_prec = latest.precision_score if latest else None
    curr_rec = latest.recall if latest else None
    curr_f1 = latest.f1 if latest else None

    task_type = model.task_type or "binary_classification"

    if task_type in ("binary_classification", "multiclass"):
        metrics = [
            _metric("Accuracy", curr_acc, base_acc),
            _metric("Precision", curr_prec, base_prec),
            _metric("Recall", curr_rec, base_rec),
            _metric("F1 Score", curr_f1, base_f1),
        ]
    else:
        # Regression metrics
        metrics = [
            _metric("MAE", latest.mae if latest else None, bv_perf.get("mae"), as_percent=False),
            _metric("RMSE", latest.rmse if latest else None, bv_perf.get("rmse"), as_percent=False),
            _metric("R²", latest.r_squared if latest else None, bv_perf.get("r2"), as_percent=False),
        ]

    time_series = []
    for s in snapshots:
        point = {
            "time": s.timestamp.strftime("%b %d") if s.timestamp else "",
            "current": round(s.accuracy, 4) if s.accuracy is not None else None,
            "base": round(base_acc, 4) if base_acc is not None else None,
        }
        time_series.append(point)

    return {
        "metrics": metrics,
        "timeSeries": time_series,
        "hasLabels": latest is not None,
        "sampleSize": latest.sample_size if latest else 0,
        "labelCoverage": latest.label_coverage if latest else None,
        "taskType": task_type,
    }


# ── Reevaluate ──────────────────────────────────────────────────────────────


@router.post("/models/{model_id}/reevaluate")
def reevaluate_model(model_id: str, db: Session = Depends(get_db)):
    """
    Trigger an immediate drift recheck using the latest predictions
    against the stored baseline.
    """
    from app.services.evaluate import evaluate_model

    model = db.query(ModelDB).filter(ModelDB.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    active_baseline = _get_active_baseline(model_id, db)
    if active_baseline:
        return evaluate_model(model, db)

    now = _now_utc()

    # Legacy path
    baselines = (
        db.query(ModelBaseline)
        .filter(ModelBaseline.model_id == model_id, ModelBaseline.feature_name != "__performance__")
        .all()
    )
    if not baselines:
        model.last_evaluated_at = now
        db.commit()
        return {
            "status": "no_baseline",
            "message": "No baseline found for this model. Upload a baseline first.",
            "model_id": model_id,
        }

    cutoff = now - timedelta(hours=24)
    predictions = (
        db.query(PredictionLog)
        .filter(PredictionLog.model_id == model_id, PredictionLog.timestamp >= cutoff)
        .order_by(PredictionLog.timestamp.desc())
        .limit(5000)
        .all()
    )

    if not predictions:
        model.last_evaluated_at = now
        db.commit()
        return {
            "status": "no_predictions",
            "message": "No recent predictions found to evaluate against.",
            "model_id": model_id,
        }

    psi_warn, psi_crit = _get_thresholds(model_id, "psi", db)
    js_warn, js_crit = settings.JS_THRESHOLD_WARNING, settings.JS_THRESHOLD_CRITICAL

    computed_drifts = []
    psi_scores: list[float] = []

    for bl in baselines:
        feature = bl.feature_name
        live_vals = []
        for pred in predictions:
            fv = pred.feature_values or {}
            if feature in fv and fv[feature] is not None:
                try:
                    live_vals.append(float(fv[feature]))
                except (ValueError, TypeError):
                    pass

        if not live_vals:
            computed_drifts.append({
                "featureName": feature,
                "status": "insufficient_data",
                "reason": "No live values found for this feature.",
                "driftScore": None,
                "jsDistance": None,
            })
            continue

        live_arr = np.array(live_vals)
        is_constant = (bl.std is not None and bl.std == 0)

        # Use histogram-based PSI
        psi_val = 0.0
        if bl.histogram_counts and bl.histogram_edges:
            edges = np.array(bl.histogram_edges)
            live_counts = np.histogram(live_arr, bins=edges)[0].astype(float)
            total_live = len(live_arr)
            if live_counts.sum() < total_live:
                live_counts[-1] += (total_live - live_counts.sum())
            psi_val = compute_psi_from_histograms(bl.histogram_counts, live_counts.tolist())

        js_val = compute_js_distance(
            baseline=np.array([bl.mean] * 100 if bl.mean is not None else [0.0]),
            current=live_arr,
        )
        ks_stat, ks_p = compute_ks_test(
            np.array([bl.mean] * 100 if bl.mean is not None else [0.0]),
            live_arr,
        )

        decision = drift_decision(
            psi=psi_val,
            js_distance=js_val,
            n_live=len(live_vals),
            n_baseline=int(sum(bl.histogram_counts)) if bl.histogram_counts else 1000,
            psi_warning=psi_warn,
            psi_critical=psi_crit,
            js_warning=js_warn,
            js_critical=js_crit,
            min_sample_size=settings.MIN_SAMPLE_SIZE,
            is_constant=is_constant,
        )

        psi_scores.append(psi_val)

        drift_rec = DriftScore(
            model_id=model_id,
            feature_name=feature,
            timestamp=now,
            psi_score=round(psi_val, 4),
            js_distance=round(js_val, 4),
            ks_statistic=round(ks_stat, 4),
            ks_pvalue=round(ks_p, 6),
            baseline_mean=bl.mean,
            current_mean=round(float(live_arr.mean()), 4),
            n_live=len(live_vals),
            n_baseline=int(sum(bl.histogram_counts)) if bl.histogram_counts else None,
            status=decision.status,
            confidence=decision.confidence,
            reason=decision.reason,
        )
        db.add(drift_rec)

        computed_drifts.append({
            "featureName": feature,
            "status": decision.status,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "driftScore": round(psi_val, 4),
            "jsDistance": round(js_val, 4),
            "nLive": len(live_vals),
        })

    avg_psi = float(np.mean(psi_scores)) if psi_scores else 0.0
    open_crits = (
        db.query(func.count(AlertDB.id))
        .filter(AlertDB.model_id == model_id, AlertDB.severity == "critical", AlertDB.status != "resolved")
        .scalar()
        or 0
    )
    health = compute_health_score(avg_psi=avg_psi, open_critical_alerts=open_crits, psi_warning=psi_warn, psi_critical=psi_crit)
    model.health_score = health.score
    model.last_evaluated_at = now
    db.commit()

    return {
        "status": "completed",
        "model_id": model_id,
        "evaluatedAt": now.isoformat(),
        "healthScore": health.score,
        "healthComponents": health.components,
        "drifts": computed_drifts,
        "nPredictions": len(predictions),
    }
