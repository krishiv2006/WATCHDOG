"""Evaluate a model against its active BaselineVersion using stored sketches.

Does not reconstruct synthetic baseline samples. PSI uses stored quantile
bucket counts; JS uses the stored fine histogram edges.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.config import get_settings
from app.models import (
    Model as ModelDB,
    BaselineVersion,
    PredictionLog,
    DriftScore,
    PerformanceSnapshot,
    Alert as AlertDB,
    utcnow,
)
from app.services.drift import (
    compute_psi_from_quantile_edges,
    compute_psi_from_histograms,
    compute_js_from_histograms,
    compute_js_from_quantile_histograms,
    compute_ks_test,
    histogram_permutation_p,
    benjamini_hochberg,
    drift_decision,
    compute_health_score,
    _clean,
)
from app.services.performance import (
    compute_classification_metrics,
    compute_regression_metrics,
    performance_ratio,
)
from app.services.alerting import (
    upsert_drift_alert,
    upsert_quality_alert,
    upsert_performance_alert,
    auto_resolve_alert,
)
from app.services.quality import volume_anomaly, feature_null_spikes

logger = logging.getLogger(__name__)
settings = get_settings()


def _get_thresholds(model_id: str, metric: str, db: Session) -> tuple[float, float]:
    from app.models import AlertRule

    rule = (
        db.query(AlertRule)
        .filter(AlertRule.model_id == model_id, AlertRule.metric == metric, AlertRule.enabled == True)  # noqa: E712
        .first()
    )
    if rule:
        return rule.threshold_warning, rule.threshold_critical
    global_rule = (
        db.query(AlertRule)
        .filter(AlertRule.model_id.is_(None), AlertRule.metric == metric, AlertRule.enabled == True)  # noqa: E712
        .first()
    )
    if global_rule:
        return global_rule.threshold_warning, global_rule.threshold_critical
    if metric == "js_distance":
        return settings.JS_THRESHOLD_WARNING, settings.JS_THRESHOLD_CRITICAL
    return settings.PSI_THRESHOLD_WARNING, settings.PSI_THRESHOLD_CRITICAL


def _get_min_sample_size(model_id: str, feature_name: str, db: Session) -> int:
    from app.models import AlertRule

    query = db.query(AlertRule).filter(
        AlertRule.metric == "psi",
        AlertRule.enabled == True,  # noqa: E712
    )
    rule = (
        query.filter(
            AlertRule.model_id == model_id,
            AlertRule.feature_name == feature_name,
        ).first()
        or query.filter(
            AlertRule.model_id == model_id,
            AlertRule.feature_name.is_(None),
        ).first()
        or query.filter(AlertRule.model_id.is_(None)).first()
    )
    return rule.min_samples if rule and rule.min_samples is not None else settings.MIN_SAMPLE_SIZE


def _live_values(predictions: list[PredictionLog], feature: str) -> list[float]:
    vals = []
    for pred in predictions:
        fv = pred.feature_values or {}
        if feature in fv and fv[feature] is not None:
            try:
                vals.append(float(fv[feature]))
            except (ValueError, TypeError):
                pass
    return vals


def _live_raw(predictions: list[PredictionLog], feature: str) -> list:
    vals = []
    for pred in predictions:
        fv = pred.feature_values or {}
        if feature in fv:
            vals.append(fv[feature])
    return vals


def prediction_window(
    db: Session,
    model_id: str,
    window_hours: int,
) -> tuple[datetime | None, datetime | None]:
    """Return a model's data-anchored evaluation window."""
    latest = (
        db.query(PredictionLog.timestamp)
        .filter(PredictionLog.model_id == model_id)
        .order_by(PredictionLog.timestamp.desc())
        .first()
    )
    if not latest or latest[0] is None:
        return None, None
    window_end = latest[0]
    if window_end.tzinfo is None:
        window_end = window_end.replace(tzinfo=utcnow().tzinfo)
    return window_end - timedelta(hours=window_hours), window_end


def _score_categorical(profile: dict, live_raw: list) -> dict:
    from collections import Counter

    freqs = profile.get("category_frequencies") or {}
    keys = list(freqs.keys())
    other_base = int(profile.get("other_count") or 0)
    baseline_counts = [int(freqs[k]) for k in keys] + [other_base]
    live_non_null = [str(v) for v in live_raw if v is not None]
    n_all = len(live_raw)
    n_live = len(live_non_null)
    null_live = 1.0 - (n_live / n_all) if n_all else 0.0
    counter = Counter(live_non_null)
    live_counts = [counter.get(k, 0) for k in keys]
    other_live = sum(c for k, c in counter.items() if k not in freqs)
    live_counts.append(other_live)
    psi_val = 0.0
    js_val = 0.0
    if n_live and any(baseline_counts):
        psi_val = compute_psi_from_histograms(baseline_counts, live_counts)
        js_val = compute_js_from_histograms(baseline_counts, live_counts)
    return {
        "psi": psi_val,
        "js": js_val,
        "n_live": n_live,
        "n_baseline": int(profile.get("count") or 0),
        "null_live": null_live,
        "null_baseline": float(profile.get("null_rate") or 0),
        "oor": 0.0,
        "current_mean": None,
        "current_std": None,
        "n_buckets": max(len(baseline_counts), 1),
        "is_constant": bool(profile.get("constant")),
        "baseline_counts": baseline_counts,
        "live_counts": live_counts,
    }


def _score_numeric(profile: dict, live_vals: list[float], n_buckets: int) -> dict:
    live_arr, null_live = _clean(np.asarray(live_vals, dtype=float), profile["name"])
    n_live = int(len(live_arr))
    n_baseline = int(profile.get("count") or 0)
    b_min = profile.get("min")
    b_max = profile.get("max")
    oor = 0.0
    if n_live and b_min is not None and b_max is not None:
        oor = float(((live_arr < b_min) | (live_arr > b_max)).mean())

    psi_val = 0.0
    edges = profile.get("quantile_edges") or []
    q_counts = profile.get("quantile_counts")
    if edges and q_counts:
        psi_val = compute_psi_from_quantile_edges(live_arr, np.array(edges), np.array(q_counts))
    elif profile.get("histogram_counts") and profile.get("histogram_edges"):
        hist_edges = np.array(profile["histogram_edges"])
        live_counts, _ = np.histogram(live_arr, bins=hist_edges)
        # Count out-of-range into outer bins
        live_counts = live_counts.astype(float)
        below = int(np.sum(live_arr < hist_edges[0])) if n_live else 0
        above = int(np.sum(live_arr > hist_edges[-1])) if n_live else 0
        if len(live_counts):
            live_counts[0] += below
            live_counts[-1] += above
        psi_val = compute_psi_from_histograms(profile["histogram_counts"], live_counts.tolist())

    js_val = 0.0
    if q_counts and n_live:
        js_val = compute_js_from_quantile_histograms(
            q_counts, live_arr, edges
        )

    return {
        "psi": psi_val,
        "js": js_val,
        "n_live": n_live,
        "n_baseline": n_baseline,
        "null_live": null_live,
        "null_baseline": float(profile.get("null_rate") or 0),
        "oor": oor,
        "current_mean": float(live_arr.mean()) if n_live else None,
        "current_std": float(live_arr.std()) if n_live > 1 else None,
        "n_buckets": n_buckets if not edges else len(edges) + 1,
        "is_constant": bool(profile.get("constant")),
        "quantile_counts": (
            np.bincount(
                np.digitize(live_arr, np.asarray(edges, dtype=float)),
                minlength=len(q_counts),
            )[: len(q_counts)].tolist()
            if q_counts and n_live else []
        ),
    }


def evaluate_model(model: ModelDB, db: Session, window_hours: int | None = None) -> dict:
    """Run drift + performance + quality for one model. Persists scores and alerts."""
    now = utcnow()
    hours = window_hours or settings.EVAL_WINDOW_HOURS
    baseline = (
        db.query(BaselineVersion)
        .filter(BaselineVersion.model_id == model.id, BaselineVersion.is_active == True)  # noqa: E712
        .first()
    )
    if not baseline:
        model.last_evaluated_at = now
        db.commit()
        return {
            "status": "no_baseline",
            "message": "No baseline found for this model. Upload a baseline first.",
            "model_id": model.id,
        }

    cutoff, window_end = prediction_window(db, model.id, hours)
    if cutoff is None or window_end is None:
        model.last_evaluated_at = now
        db.commit()
        return {
            "status": "no_predictions",
            "message": "No predictions found to evaluate against.",
            "model_id": model.id,
        }

    predictions = (
        db.query(PredictionLog)
        .filter(
            PredictionLog.model_id == model.id,
            PredictionLog.timestamp >= cutoff,
            PredictionLog.timestamp <= window_end,
        )
        .order_by(PredictionLog.timestamp.desc())
        .limit(settings.MAX_BATCH_PREDICTIONS)
        .all()
    )
    if not predictions:
        model.last_evaluated_at = now
        db.commit()
        return {
            "status": "no_predictions",
            "message": "No recent predictions found to evaluate against.",
            "model_id": model.id,
        }

    # Re-evaluating the same live window must replace, not duplicate, scores.
    db.query(DriftScore).filter(
        DriftScore.model_id == model.id,
        DriftScore.timestamp >= cutoff,
        DriftScore.timestamp <= window_end,
    ).delete(synchronize_session=False)
    db.flush()

    psi_warn, psi_crit = _get_thresholds(model.id, "psi", db)
    js_warn, js_crit = _get_thresholds(model.id, "js_distance", db)
    model_name = f"{model.name} {model.version}"
    computed = []
    psi_scores: list[float] = []
    scored_cache: dict[str, dict] = {}
    raw_p_values: dict[str, float] = {}
    ks_cache: dict[str, tuple[float | None, float]] = {}
    for profile in baseline.feature_profiles or []:
        feature = profile["name"]
        if profile.get("constant"):
            continue
        if profile.get("feature_type") == "categorical":
            scored = _score_categorical(profile, _live_raw(predictions, feature))
        else:
            scored = _score_numeric(
                profile,
                _live_values(predictions, feature),
                baseline.n_psi_buckets or 10,
            )
        scored_cache[feature] = scored
        if profile.get("feature_type") == "categorical":
            raw_p_values[feature] = histogram_permutation_p(
                scored["baseline_counts"], scored["live_counts"],
                settings.DRIFT_PERMUTATION_RESAMPLES,
            )
            ks_cache[feature] = (None, raw_p_values[feature])
        else:
            edges = np.asarray(profile.get("quantile_edges") or [], dtype=float)
            counts = np.asarray(profile.get("quantile_counts") or [], dtype=int)
            boundaries = np.concatenate((
                [float(profile["min"])], edges, [float(profile["max"])]
            )) if len(edges) else np.asarray([], dtype=float)
            if len(counts) and len(boundaries) == len(counts) + 1:
                centers = (boundaries[:-1] + boundaries[1:]) / 2
                baseline_values = np.repeat(centers, counts)
                ks_stat, ks_p = compute_ks_test(baseline_values, _live_values(predictions, feature))
            else:
                ks_stat, ks_p = 0.0, 1.0
            raw_p_values[feature] = ks_p
            ks_cache[feature] = (ks_stat, ks_p)
    adjusted_p_values = benjamini_hochberg(raw_p_values)

    for profile in baseline.feature_profiles or []:
        feature = profile["name"]
        if profile.get("constant"):
            computed.append({
                "featureName": feature,
                "status": "insufficient_data",
                "reason": "Constant baseline column — drift undefined.",
                "driftScore": None,
                "jsDistance": None,
            })
            continue

        scored = scored_cache.get(feature)
        if scored is None:
            continue
        decision = drift_decision(
            psi=scored["psi"],
            js_distance=scored["js"],
            n_live=scored["n_live"],
            n_baseline=scored["n_baseline"],
            n_buckets=scored["n_buckets"],
            psi_warning=psi_warn,
            psi_critical=psi_crit,
            js_warning=js_warn,
            js_critical=js_crit,
            min_sample_size=_get_min_sample_size(model.id, feature, db),
            is_constant=scored["is_constant"],
            ks_p=adjusted_p_values.get(feature),
        )
        if scored["n_live"] > 0 and decision.status != "insufficient_data":
            psi_scores.append(scored["psi"])

        rec = DriftScore(
            model_id=model.id,
            feature_name=feature,
            timestamp=now,
            psi_score=round(scored["psi"], 4),
            js_distance=round(scored["js"], 4),
            baseline_mean=profile.get("mean"),
            current_mean=round(scored["current_mean"], 4) if scored["current_mean"] is not None else None,
            baseline_std=profile.get("std"),
            current_std=round(scored["current_std"], 4) if scored["current_std"] is not None else None,
            ks_statistic=round(ks_cache[feature][0], 4) if ks_cache[feature][0] is not None else None,
            ks_pvalue=round(adjusted_p_values.get(feature, ks_cache[feature][1]), 6),
            n_live=scored["n_live"],
            n_baseline=scored["n_baseline"],
            null_rate_baseline=scored["null_baseline"],
            null_rate_live=round(scored["null_live"], 4),
            out_of_range_rate=round(scored["oor"], 4),
            status=decision.status,
            confidence=decision.confidence,
            reason=decision.reason,
            baseline_version_id=baseline.id,
        )
        db.add(rec)
        computed.append({
            "featureName": feature,
            "status": decision.status,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "driftScore": round(scored["psi"], 4),
            "jsDistance": round(scored["js"], 4),
            "nLive": scored["n_live"],
            "nBaseline": scored["n_baseline"],
            "outOfRangeRate": round(scored["oor"], 4),
        })
        upsert_drift_alert(
            db,
            model.id,
            model_name,
            feature,
            decision.status,
            scored["psi"],
            scored["js"],
            scored["n_live"],
            decision.reason,
            profile.get("mean"),
            scored["current_mean"],
        )

    # Volume anomaly
    vol = volume_anomaly(db, model.id, window_hours=hours)
    if vol:
        upsert_quality_alert(
            db, model.id, model_name, "volume",
            "Input Volume Anomaly",
            f"Last window ingested {vol['current']} predictions vs rolling average {vol['expected']}.",
            vol["ratio"],
        )
    else:
        auto_resolve_alert(
            db, model.id, f"{model.id}:data_quality:volume", model_name
        )

    spikes = feature_null_spikes(predictions, baseline)
    spike_fingerprints = set()
    for sp in spikes:
        spike_fingerprints.add(f"{model.id}:data_quality:null:{sp['feature']}")
        upsert_quality_alert(
            db, model.id, model_name, f"null:{sp['feature']}",
            f"Null-rate spike: {sp['feature']}",
            f"Live null rate {sp['liveNullRate']} vs baseline {sp['baselineNullRate']}.",
            sp["liveNullRate"],
        )
    for profile in baseline.feature_profiles or []:
        fingerprint = f"{model.id}:data_quality:null:{profile['name']}"
        if fingerprint not in spike_fingerprints:
            auto_resolve_alert(
                db, model.id, fingerprint, model_name,
                "Auto-resolved: the column is no longer an eligible live feature.",
            )

    target_column = (
        (baseline.prediction_distribution or {}).get("target_column")
        if baseline.prediction_distribution
        else None
    )
    active_features = {
        profile["name"] for profile in baseline.feature_profiles or []
        if profile["name"] != target_column
    }
    open_quality_alerts = (
        db.query(AlertDB)
        .filter(
            AlertDB.model_id == model.id,
            AlertDB.category == "data_quality",
            AlertDB.status != "resolved",
            AlertDB.fingerprint.like(f"{model.id}:data_quality:null:%"),
        )
        .all()
    )
    for alert in open_quality_alerts:
        feature = alert.fingerprint.rsplit(":", 1)[-1]
        if feature not in active_features:
            auto_resolve_alert(
                db, model.id, alert.fingerprint, model_name,
                "Auto-resolved: the column is not an active baseline feature.",
            )

    open_drift_alerts = (
        db.query(AlertDB)
        .filter(
            AlertDB.model_id == model.id,
            AlertDB.category == "data_drift",
            AlertDB.status != "resolved",
        )
        .all()
    )
    for alert in open_drift_alerts:
        feature = alert.fingerprint.rsplit(":", 1)[-1]
        if feature not in active_features:
            auto_resolve_alert(
                db, model.id, alert.fingerprint, model_name,
                "Auto-resolved: the column is not an active baseline feature.",
            )

    # Performance from labels only
    labeled = [p for p in predictions if p.actual_value is not None and p.predicted_value is not None]
    n_pred = len(predictions)
    n_lab = len(labeled)
    coverage = n_lab / n_pred if n_pred else 0.0
    perf_ratio = None
    snap = PerformanceSnapshot(
        model_id=model.id,
        timestamp=now,
        sample_size=n_pred,
        n_labeled=n_lab,
        label_coverage=round(coverage, 4),
        perf_status="awaiting_labels",
    )
    min_labels = settings.MIN_LABELS
    if n_lab >= min_labels:
        y_true = np.array([p.actual_value for p in labeled], dtype=float)
        y_pred = np.array([p.predicted_value for p in labeled], dtype=float)
        scores = np.array([p.score if p.score is not None else p.predicted_value for p in labeled], dtype=float)
        task = model.task_type or "binary_classification"
        if task == "regression":
            m = compute_regression_metrics(y_true, y_pred)
            snap.mae = m.get("mae")
            snap.rmse = m.get("rmse")
            snap.r_squared = m.get("r2")
            snap.perf_status = "healthy"
            base_perf = (baseline.performance_metrics or {})
            perf_ratio = performance_ratio(m.get("rmse"), base_perf.get("rmse"), higher_is_better=False)
        else:
            m = compute_classification_metrics(
                np.round(y_true), np.round(y_pred), scores,
                multiclass=(task == "multiclass"),
            )
            snap.accuracy = m.get("accuracy")
            snap.precision_score = m.get("precision")
            snap.recall = m.get("recall")
            snap.f1 = m.get("f1")
            snap.auc = m.get("auc")
            snap.perf_status = "healthy"
            base_perf = (baseline.performance_metrics or {})
            perf_ratio = performance_ratio(m.get("accuracy"), base_perf.get("accuracy"), higher_is_better=True)
            if perf_ratio is not None and perf_ratio < 0.9:
                snap.perf_status = "degraded"
                upsert_performance_alert(
                    db, model.id, model_name,
                    f"Live accuracy/error is {perf_ratio:.0%} of baseline performance.",
                    perf_ratio,
                )
            else:
                auto_resolve_alert(
                    db, model.id, f"{model.id}:performance:degradation", model_name
                )
    else:
        auto_resolve_alert(
            db, model.id, f"{model.id}:performance:degradation", model_name
        )
    db.add(snap)

    avg_psi = float(np.mean(psi_scores)) if psi_scores else 0.0
    open_crits = (
        db.query(func.count(AlertDB.id))
        .filter(AlertDB.model_id == model.id, AlertDB.severity == "critical", AlertDB.status != "resolved")
        .scalar()
        or 0
    )
    health = None
    if psi_scores:
        health = compute_health_score(
            avg_psi=avg_psi,
            open_critical_alerts=open_crits,
            psi_warning=psi_warn,
            psi_critical=psi_crit,
            performance_ratio=perf_ratio,
            weight_drift=settings.HEALTH_WEIGHT_DRIFT,
            weight_perf=settings.HEALTH_WEIGHT_PERFORMANCE,
            weight_alerts=settings.HEALTH_WEIGHT_ALERTS,
            alert_penalty=settings.HEALTH_ALERT_PENALTY,
            drift_exponent=settings.HEALTH_DRIFT_EXPONENT,
        )
    model.health_score = health.score if health else None
    model.health_components = health.components if health else None
    model.last_evaluated_at = now
    db.commit()
    return {
        "status": "completed",
        "model_id": model.id,
        "evaluatedAt": now.isoformat(),
        "healthScore": health.score if health else None,
        "healthComponents": health.components if health else None,
        "drifts": computed,
        "nPredictions": n_pred,
        "nLabeled": n_lab,
        "labelCoverage": round(coverage, 4),
        "baselineVersion": baseline.version_number,
        "perfStatus": snap.perf_status,
    }
