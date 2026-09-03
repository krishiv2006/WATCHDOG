"""
Seed script that creates demo models with baselines and simulates
30 days of drifting prediction data for a fully demoable system.

Usage:
    python -m scripts.seed_demo_data
"""

import sys
import os

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timedelta
from app.database import engine, SessionLocal, Base
from app.models import (
    Model, ModelBaseline, PredictionLog, DriftScore,
    PerformanceSnapshot, Alert, AlertRule, Report
)
from app.services.drift import compute_psi, compute_js_distance, psi_to_status, compute_health_score

rng = np.random.default_rng(42)


# ── Demo model definitions ─────────────────────────
DEMO_MODELS = [
    {
        "id": "m-1",
        "name": "Customer Churn Predictor",
        "version": "v2.1",
        "algorithm": "XGBoost",
        "environment": "production",
        "status": "production",
        "training_data_ref": "customer_base_2023_q4.csv",
        "features": {
            "annual_income": {"baseline_mean": 65000, "baseline_std": 15000, "drift_shift": 7000, "drift_std_mult": 1.1},
            "usage_frequency": {"baseline_mean": 14.5, "baseline_std": 5.0, "drift_shift": -2.4, "drift_std_mult": 1.2},
            "customer_tenure": {"baseline_mean": 36.2, "baseline_std": 12.0, "drift_shift": -0.4, "drift_std_mult": 1.0},
            "support_tickets": {"baseline_mean": 2.1, "baseline_std": 1.5, "drift_shift": 0.8, "drift_std_mult": 1.3},
            "contract_length": {"baseline_mean": 18.0, "baseline_std": 6.0, "drift_shift": -1.0, "drift_std_mult": 1.0},
        },
        "baseline_perf": {"accuracy": 0.942, "precision": 0.933, "recall": 0.939, "f1": 0.936},
        "current_perf_degradation": 0.048,  # accuracy drops ~4.8%
    },
    {
        "id": "m-2",
        "name": "Fraud Detection Engine",
        "version": "v1.4",
        "algorithm": "Random Forest",
        "environment": "production",
        "status": "retraining",
        "training_data_ref": "fraud_transactions_2024_h1.csv",
        "features": {
            "transaction_amount": {"baseline_mean": 250, "baseline_std": 180, "drift_shift": 80, "drift_std_mult": 1.4},
            "merchant_risk_score": {"baseline_mean": 0.35, "baseline_std": 0.2, "drift_shift": 0.15, "drift_std_mult": 1.3},
            "time_since_last_tx": {"baseline_mean": 48, "baseline_std": 30, "drift_shift": -10, "drift_std_mult": 1.1},
            "geo_distance": {"baseline_mean": 15, "baseline_std": 20, "drift_shift": 5, "drift_std_mult": 1.2},
        },
        "baseline_perf": {"accuracy": 0.965, "precision": 0.958, "recall": 0.948, "f1": 0.953},
        "current_perf_degradation": 0.085,
    },
    {
        "id": "m-3",
        "name": "Product Recommender",
        "version": "v3.0",
        "algorithm": "Neural Collaborative Filtering",
        "environment": "production",
        "status": "production",
        "training_data_ref": "user_interactions_2024_q2.csv",
        "features": {
            "user_activity_score": {"baseline_mean": 7.2, "baseline_std": 2.5, "drift_shift": 0.3, "drift_std_mult": 1.0},
            "purchase_history_length": {"baseline_mean": 24, "baseline_std": 15, "drift_shift": 2, "drift_std_mult": 1.05},
            "session_duration": {"baseline_mean": 12.5, "baseline_std": 8, "drift_shift": -0.5, "drift_std_mult": 1.0},
            "category_diversity": {"baseline_mean": 5.8, "baseline_std": 2.0, "drift_shift": 0.2, "drift_std_mult": 1.0},
        },
        "baseline_perf": {"accuracy": 0.912, "precision": 0.905, "recall": 0.898, "f1": 0.901},
        "current_perf_degradation": 0.012,
    },
    {
        "id": "m-4",
        "name": "Credit Risk Scorer",
        "version": "v1.0",
        "algorithm": "LightGBM",
        "environment": "staging",
        "status": "staging",
        "training_data_ref": "credit_applications_2024.csv",
        "features": {
            "debt_to_income": {"baseline_mean": 0.35, "baseline_std": 0.15, "drift_shift": 0.02, "drift_std_mult": 1.0},
            "credit_utilization": {"baseline_mean": 0.45, "baseline_std": 0.25, "drift_shift": 0.05, "drift_std_mult": 1.1},
            "payment_history": {"baseline_mean": 0.85, "baseline_std": 0.1, "drift_shift": -0.03, "drift_std_mult": 1.0},
        },
        "baseline_perf": {"accuracy": 0.928, "precision": 0.915, "recall": 0.920, "f1": 0.917},
        "current_perf_degradation": 0.025,
    },
]


def seed_database():
    """Drop all tables and re-create with fresh demo data."""
    print("Dropping existing tables...")
    Base.metadata.drop_all(bind=engine)

    print("Creating tables...")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    now = datetime.utcnow()

    try:
        for model_def in DEMO_MODELS:
            print(f"Seeding model: {model_def['name']} ({model_def['version']})")

            # ── Create model ───────────────────────────
            model = Model(
                id=model_def["id"],
                name=model_def["name"],
                version=model_def["version"],
                algorithm=model_def["algorithm"],
                environment=model_def["environment"],
                status=model_def["status"],
                training_data_ref=model_def["training_data_ref"],
                created_at=now - timedelta(days=90),
                last_evaluated_at=now - timedelta(minutes=10),
            )
            db.add(model)

            # ── Create feature baselines ───────────────
            n_baseline = 2000
            for feat_name, feat_cfg in model_def["features"].items():
                baseline_data = rng.normal(feat_cfg["baseline_mean"], feat_cfg["baseline_std"], n_baseline)
                hist_counts, hist_edges = np.histogram(baseline_data, bins=10)
                histogram_counts_list = [int(x) for x in hist_counts]
                histogram_edges_list = [float(x) for x in hist_edges]

                baseline = ModelBaseline(
                    model_id=model_def["id"],
                    feature_name=feat_name,
                    mean=float(feat_cfg["baseline_mean"]),
                    std=float(feat_cfg["baseline_std"]),
                    histogram_counts=histogram_counts_list,
                    histogram_edges=histogram_edges_list,
                )
                db.add(baseline)

            # ── Create performance baseline ────────────
            perf = model_def["baseline_perf"]
            perf_baseline = ModelBaseline(
                model_id=model_def["id"],
                feature_name="__performance__",
                mean=0, std=0,
                accuracy=perf["accuracy"],
                precision_score=perf["precision"],
                recall=perf["recall"],
                f1=perf["f1"],
            )
            db.add(perf_baseline)

            # ── Generate 30 days of prediction logs ────
            days = 30
            predictions_per_day = 200
            total_predictions = 0

            for day_offset in range(days, 0, -1):
                ts_base = now - timedelta(days=day_offset)
                # Drift increases linearly over 30 days
                drift_fraction = (days - day_offset) / days

                for _ in range(predictions_per_day):
                    ts = ts_base + timedelta(
                        hours=int(rng.integers(0, 24)),
                        minutes=int(rng.integers(0, 60)),
                    )

                    feature_values = {}
                    for feat_name, feat_cfg in model_def["features"].items():
                        shifted_mean = feat_cfg["baseline_mean"] + feat_cfg["drift_shift"] * drift_fraction
                        shifted_std = feat_cfg["baseline_std"] * (1 + (feat_cfg["drift_std_mult"] - 1) * drift_fraction)
                        val = rng.normal(shifted_mean, shifted_std)
                        feature_values[feat_name] = float(val)

                    # Simulate prediction + ground truth
                    pred = float(1.0 if rng.random() > 0.5 else 0.0)
                    degradation = float(model_def["current_perf_degradation"] * drift_fraction)
                    correct = bool(rng.random() > degradation)
                    actual = float(pred if correct else (1.0 - pred))

                    prediction = PredictionLog(
                        model_id=model_def["id"],
                        timestamp=ts,
                        feature_values=feature_values,
                        predicted_value=pred,
                        actual_value=actual,
                        latency_ms=float(rng.normal(45, 15)),
                    )
                    db.add(prediction)
                    total_predictions += 1

            print(f"   ✅ {total_predictions} prediction logs created")

            # ── Compute & store drift scores ───────────
            n_current = 500  # recent samples for drift calc
            latest_drift_scores = []
            for feat_name, feat_cfg in model_def["features"].items():
                baseline_data = rng.normal(feat_cfg["baseline_mean"], feat_cfg["baseline_std"], n_baseline)
                current_data = rng.normal(
                    feat_cfg["baseline_mean"] + feat_cfg["drift_shift"],
                    feat_cfg["baseline_std"] * feat_cfg["drift_std_mult"],
                    n_current,
                )

                psi = compute_psi(baseline_data, current_data)
                js = compute_js_distance(baseline_data, current_data)
                status = psi_to_status(psi)

                drift = DriftScore(
                    model_id=model_def["id"],
                    feature_name=feat_name,
                    timestamp=now,
                    psi_score=float(round(psi, 4)),
                    js_distance=float(round(js, 4)),
                    baseline_mean=float(feat_cfg["baseline_mean"]),
                    current_mean=float(round(feat_cfg["baseline_mean"] + feat_cfg["drift_shift"], 2)),
                    status=status,
                )
                db.add(drift)
                latest_drift_scores.append(float(psi))
                print(f"   📊 {feat_name}: PSI={psi:.4f} JS={js:.4f} [{status}]")

            # ── Generate performance snapshots over 30 days ──
            for day_offset in range(30, 0, -1):
                ts = now - timedelta(days=day_offset)
                drift_fraction = (30 - day_offset) / 30
                deg = model_def["current_perf_degradation"] * drift_fraction

                snapshot = PerformanceSnapshot(
                    model_id=model_def["id"],
                    timestamp=ts,
                    accuracy=float(round(perf["accuracy"] - deg, 4)),
                    precision_score=float(round(perf["precision"] - deg * 0.5, 4)),
                    recall=float(round(perf["recall"] - deg * 1.2, 4)),
                    f1=float(round(perf["f1"] - deg * 0.8, 4)),
                    sample_size=int(predictions_per_day),
                )
                db.add(snapshot)

            # ── Calculate health score ─────────────────
            avg_drift = float(np.mean(latest_drift_scores)) if latest_drift_scores else 0.0
            current_accuracy = perf["accuracy"] - model_def["current_perf_degradation"]
            perf_ratio = current_accuracy / perf["accuracy"]
            open_critical = sum(1 for d in latest_drift_scores if d > 0.25)
            health = float(compute_health_score(avg_drift, perf_ratio, open_critical))
            model.health_score = health
            print(f"   💊 Health Score: {health}")

            # ── Default alert rules ────────────────────
            for metric, warn, crit in [
                ("psi", 0.1, 0.25),
                ("js_distance", 0.15, 0.35),
                ("accuracy_delta", -3.0, -5.0),
                ("latency_p99", 200, 500),
            ]:
                rule = AlertRule(
                    model_id=model_def["id"],
                    metric=metric,
                    threshold_warning=warn,
                    threshold_critical=crit,
                    enabled=True,
                )
                db.add(rule)

        # ── Seed some alerts ───────────────────────────
        print("\n🚨 Seeding alerts...")
        alerts_data = [
            {
                "model_id": "m-1", "severity": "warning", "category": "data_drift",
                "title": "Data Drift Exceeded PSI Threshold",
                "description": "Feature 'annual_income' drifted beyond 0.25 (Current: 0.31)",
                "detected_at": now - timedelta(minutes=2), "status": "unacknowledged",
            },
            {
                "model_id": "m-2", "severity": "critical", "category": "performance",
                "title": "Performance Degradation",
                "description": "F1 Score dropped by 5.2% against baseline (Current: 82.1%)",
                "detected_at": now - timedelta(minutes=15), "status": "investigating",
            },
            {
                "model_id": "m-1", "severity": "critical", "category": "prediction_drift",
                "title": "Prediction Drift Exceeded Threshold",
                "description": "Prediction distribution shifted significantly (JS Distance: 0.42)",
                "detected_at": now - timedelta(hours=1), "status": "unacknowledged",
            },
            {
                "model_id": "m-3", "severity": "warning", "category": "data_drift",
                "title": "Feature Distribution Shift Detected",
                "description": "Feature 'purchase_history_length' showing moderate drift (PSI: 0.19)",
                "detected_at": now - timedelta(minutes=25), "status": "unacknowledged",
            },
            {
                "model_id": "m-1", "severity": "warning", "category": "data_drift",
                "title": "Categorical Feature Drift",
                "description": "Feature 'region_code' distribution changed significantly",
                "detected_at": now - timedelta(minutes=45), "status": "investigating",
            },
            {
                "model_id": "m-2", "severity": "critical", "category": "latency",
                "title": "Inference Latency Spike",
                "description": "P99 latency exceeded 500ms threshold (Current: 782ms)",
                "detected_at": now - timedelta(hours=1), "status": "unacknowledged",
            },
            {
                "model_id": "m-1", "severity": "warning", "category": "performance",
                "title": "Precision Below Warning Threshold",
                "description": "Model precision dropped to 88.1%, below 90% warning threshold",
                "detected_at": now - timedelta(hours=2), "status": "unacknowledged",
            },
            {
                "model_id": "m-3", "severity": "warning", "category": "data_drift",
                "title": "Input Volume Anomaly",
                "description": "Prediction volume 40% above rolling 7-day average",
                "detected_at": now - timedelta(hours=3), "status": "unacknowledged",
            },
            {
                "model_id": "m-2", "severity": "warning", "category": "prediction_drift",
                "title": "Prediction Confidence Shift",
                "description": "Mean prediction confidence dropped from 0.82 to 0.74",
                "detected_at": now - timedelta(hours=4), "status": "investigating",
            },
            {
                "model_id": "m-1", "severity": "warning", "category": "data_drift",
                "title": "Support Tickets Feature Drift",
                "description": "Feature 'support_tickets' PSI increased to 0.22",
                "detected_at": now - timedelta(hours=5), "status": "resolved",
                "resolved_at": now - timedelta(hours=3),
            },
            {
                "model_id": "m-4", "severity": "warning", "category": "data_drift",
                "title": "Credit Utilization Shift",
                "description": "Feature 'credit_utilization' showing upward trend (PSI: 0.14)",
                "detected_at": now - timedelta(hours=6), "status": "unacknowledged",
            },
            {
                "model_id": "m-2", "severity": "critical", "category": "performance",
                "title": "Recall Below Critical Threshold",
                "description": "Model recall dropped to 86.2%, below 88% critical threshold",
                "detected_at": now - timedelta(hours=7), "status": "resolved",
                "resolved_at": now - timedelta(hours=2),
            },
        ]
        for alert_dict in alerts_data:
            ad = dict(alert_dict)
            resolved_at = ad.pop("resolved_at", None)
            alert = Alert(**ad)
            if resolved_at:
                alert.resolved_at = resolved_at
            db.add(alert)
            print(f"   🔔 [{ad['severity'].upper()}] {ad['title']}")

        # ── Seed reports ───────────────────────────────
        print("\n📋 Seeding reports...")
        reports_data = [
            {"model_id": "m-1", "name": "Monthly Drift Summary", "report_type": "drift_analysis",
             "date_range_start": now - timedelta(days=30), "date_range_end": now,
             "status": "ready", "generated_at": now - timedelta(hours=2)},
            {"model_id": "m-2", "name": "Performance Baseline Comparison", "report_type": "performance",
             "date_range_start": now - timedelta(days=30), "date_range_end": now,
             "status": "ready", "generated_at": now - timedelta(days=1)},
            {"model_id": "m-3", "name": "Feature Drift Analysis", "report_type": "drift_analysis",
             "date_range_start": now - timedelta(days=30), "date_range_end": now,
             "status": "generating", "generated_at": None},
            {"model_id": "m-1", "name": "Quarterly Model Health", "report_type": "comprehensive",
             "date_range_start": now - timedelta(days=90), "date_range_end": now,
             "status": "failed", "generated_at": now - timedelta(hours=5)},
            {"model_id": "m-1", "name": "Alert Frequency Report", "report_type": "comprehensive",
             "date_range_start": now - timedelta(days=30), "date_range_end": now,
             "status": "ready", "generated_at": now - timedelta(days=3)},
        ]
        for rd in reports_data:
            report = Report(**rd)
            db.add(report)
            print(f"   📄 {rd['name']} [{rd['status']}]")

        db.commit()
        print("\n✅ Database seeded successfully!")
        print(f"   Models: {len(DEMO_MODELS)}")
        print(f"   Alerts: {len(alerts_data)}")
        print(f"   Reports: {len(reports_data)}")

    except Exception as e:
        db.rollback()
        import traceback
        traceback.print_exc()
        print(f"\n❌ Error seeding database: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
