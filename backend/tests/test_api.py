"""
API integration tests using FastAPI TestClient with an isolated in-memory DB.

Covers:
- Model CRUD
- Baseline upload (CSV)
- Batch prediction ingestion + idempotency
- Ground-truth join
- Reevaluate endpoint (real drift run)
- Alert lifecycle (create, acknowledge, resolve)
- Alert rules CRUD + camelCase contract
- Dashboard summary
- Window parsing validation
- Reports generation
"""

from __future__ import annotations

import io
import csv
import json
import pytest
import numpy as np
from datetime import timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
import app.models  # noqa: F401  # Ensure ORM metadata exists before create_all()
from app.main import app
from app.config import get_settings
from app.models import BaselineVersion, DriftScore, utcnow


# ── Isolated test DB ────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def db_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_engine):
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    # Disable scheduler for tests
    import app.services.scheduler as sched_mod
    original_enable = get_settings().ENABLE_SCHEDULER
    get_settings().ENABLE_SCHEDULER = False

    with TestClient(app, raise_server_exceptions=True) as c:
        # Expose a helper to get a direct DB session for test setup
        c._test_session = TestingSession
        yield c

    app.dependency_overrides.clear()
    get_settings().ENABLE_SCHEDULER = original_enable


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_baseline_csv(n: int = 500, shift: float = 0.0) -> bytes:
    """Generate a simple CSV with numeric features."""
    rng = np.random.default_rng(42)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["age", "income", "credit_score"])
    for _ in range(n):
        writer.writerow([
            round(float(rng.normal(35 + shift, 8)), 2),
            round(float(rng.normal(50000 + shift * 1000, 15000)), 2),
            round(float(rng.normal(700 + shift, 50)), 2),
        ])
    return buf.getvalue().encode()


def _create_model(client) -> str:
    resp = client.post("/api/models", json={
        "name": "Test Model",
        "version": "v1.0",
        "algorithm": "XGBoost",
        "environment": "staging",
        "taskType": "binary_classification",
    })
    assert resp.status_code == 201
    return resp.json()["id"]


def _upload_baseline(client, model_id: str, n: int = 500) -> dict:
    csv_bytes = _make_baseline_csv(n)
    resp = client.post(
        f"/api/models/{model_id}/baseline",
        files={"file": ("baseline.csv", csv_bytes, "text/csv")},
        data={"n_buckets": "10"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ── Model CRUD ───────────────────────────────────────────────────────────────

class TestModelCRUD:
    def test_create_and_list(self, client):
        mid = _create_model(client)
        resp = client.get("/api/models")
        assert resp.status_code == 200
        ids = [m["id"] for m in resp.json()]
        assert mid in ids

    def test_get_model(self, client):
        mid = _create_model(client)
        resp = client.get(f"/api/models/{mid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == mid
        assert data["name"] == "Test Model"
        assert "healthScore" in data

    def test_get_nonexistent_model_404(self, client):
        resp = client.get("/api/models/does-not-exist")
        assert resp.status_code == 404

    def test_model_has_camel_case_keys(self, client):
        mid = _create_model(client)
        resp = client.get(f"/api/models/{mid}")
        data = resp.json()
        assert "healthScore" in data
        assert "lastUpdated" in data
        assert "taskType" in data


# ── Baseline upload ──────────────────────────────────────────────────────────

class TestBaselineUpload:
    def test_upload_creates_baseline_version(self, client):
        mid = _create_model(client)
        result = _upload_baseline(client, mid)
        assert result["status"] == "success"
        assert result["featureCount"] == 3
        assert result["sampleSize"] == 500
        assert "baselineVersionId" in result

    def test_upload_stores_distribution_profile_without_performance_fabrication(self, client):
        mid = _create_model(client)
        _upload_baseline(client, mid)
        db_session = next(client.app.dependency_overrides[get_db]())
        baseline = db_session.query(BaselineVersion).filter(
            BaselineVersion.model_id == mid
        ).one()
        profile = baseline.feature_profiles[0]
        assert profile["type"] == "numeric"
        assert {"null_rate", "min", "max", "mean", "std"} <= profile.keys()
        assert profile["quantile_edges"]
        assert profile["histogram_counts"]
        assert profile["histogram_edges"]
        assert baseline.performance_metrics is None

    def test_upload_lists_in_baselines(self, client):
        mid = _create_model(client)
        _upload_baseline(client, mid)
        resp = client.get(f"/api/models/{mid}/baselines")
        assert resp.status_code == 200
        versions = resp.json()
        assert len(versions) == 1
        assert versions[0]["isActive"] is True
        assert versions[0]["featureCount"] == 3

    def test_second_upload_deactivates_first(self, client):
        mid = _create_model(client)
        _upload_baseline(client, mid, n=300)
        _upload_baseline(client, mid, n=400)
        resp = client.get(f"/api/models/{mid}/baselines")
        versions = resp.json()
        assert len(versions) == 2
        active = [v for v in versions if v["isActive"]]
        assert len(active) == 1
        assert active[0]["sampleSize"] == 400

    def test_upload_nonexistent_model_404(self, client):
        csv_bytes = _make_baseline_csv(100)
        resp = client.post(
            "/api/models/no-such-model/baseline",
            files={"file": ("b.csv", csv_bytes, "text/csv")},
            data={"n_buckets": "10"},
        )
        assert resp.status_code == 404

    def test_empty_csv_rejected(self, client):
        mid = _create_model(client)
        resp = client.post(
            f"/api/models/{mid}/baseline",
            files={"file": ("empty.csv", b"age,income\n", "text/csv")},
            data={"n_buckets": "10"},
        )
        assert resp.status_code == 400


# ── Prediction ingestion ─────────────────────────────────────────────────────

class TestPredictionIngestion:
    def _make_batch(self, n: int = 10, start: int = 0) -> dict:
        rng = np.random.default_rng(start)
        return {
            "predictions": [
                {
                    "predictionId": f"pred-{start + i:05d}",
                    "features": {
                        "age": round(float(rng.normal(35, 8)), 2),
                        "income": round(float(rng.normal(50000, 15000)), 2),
                        "credit_score": round(float(rng.normal(700, 50)), 2),
                    },
                    "prediction": float(rng.integers(0, 2)),
                    "score": round(float(rng.uniform(0, 1)), 4),
                    "latencyMs": round(float(rng.uniform(5, 50)), 2),
                }
                for i in range(n)
            ]
        }

    def test_ingest_batch(self, client):
        mid = _create_model(client)
        batch = self._make_batch(20)
        resp = client.post(f"/api/models/{mid}/predictions", json=batch)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ingested"] == 20
        assert data["duplicates"] == 0

    def test_ingest_single_prediction(self, client):
        mid = _create_model(client)
        response = client.post(
            f"/api/models/{mid}/predictions",
            json={
                "predictionId": "single-1",
                "features": {"age": 30.0},
                "prediction": 1.0,
            },
        )
        assert response.status_code == 200
        assert response.json()["ingested"] == 1

    def test_duplicate_prediction_ids_in_same_batch_are_ignored(self, client):
        mid = _create_model(client)
        response = client.post(
            f"/api/models/{mid}/predictions",
            json={
                "predictions": [
                    {"predictionId": "same", "features": {"x": 1.0}, "prediction": 0.0},
                    {"predictionId": "same", "features": {"x": 2.0}, "prediction": 1.0},
                ]
            },
        )
        assert response.status_code == 200
        assert response.json()["ingested"] == 1
        assert response.json()["duplicates"] == 1

    def test_idempotency_on_prediction_id(self, client):
        mid = _create_model(client)
        batch = self._make_batch(5)
        client.post(f"/api/models/{mid}/predictions", json=batch)
        resp2 = client.post(f"/api/models/{mid}/predictions", json=batch)
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["ingested"] == 0
        assert data["duplicates"] == 5

    def test_partial_duplicate(self, client):
        mid = _create_model(client)
        batch1 = self._make_batch(5, start=0)
        batch2 = self._make_batch(5, start=3)   # 2 overlap, 3 new
        client.post(f"/api/models/{mid}/predictions", json=batch1)
        resp = client.post(f"/api/models/{mid}/predictions", json=batch2)
        data = resp.json()
        assert data["ingested"] == 3
        assert data["duplicates"] == 2

    def test_batch_too_large_rejected(self, client):
        mid = _create_model(client)
        settings = get_settings()
        big_batch = {
            "predictions": [
                {"features": {"age": 30.0}, "prediction": 0.0}
                for _ in range(settings.MAX_BATCH_PREDICTIONS + 1)
            ]
        }
        resp = client.post(f"/api/models/{mid}/predictions", json=big_batch)
        assert resp.status_code == 400

    def test_ingest_nonexistent_model_404(self, client):
        resp = client.post(
            "/api/models/no-such/predictions",
            json={"predictions": [{"features": {"x": 1.0}, "prediction": 0.0}]},
        )
        assert resp.status_code == 404

    def test_legacy_dataset_ingestion_is_removed(self, client):
        response = client.post(
            "/api/models/ingest-dataset",
            files={"file": ("training.csv", _make_baseline_csv(), "text/csv")},
        )
        assert response.status_code == 410


# ── Ground truth ─────────────────────────────────────────────────────────────

class TestGroundTruth:
    def test_label_join(self, client):
        mid = _create_model(client)
        batch = {
            "predictions": [
                {"predictionId": f"p-{i}", "features": {"age": 30.0}, "prediction": float(i % 2)}
                for i in range(5)
            ]
        }
        client.post(f"/api/models/{mid}/predictions", json=batch)

        labels = {
            "labels": [
                {"predictionId": f"p-{i}", "actualValue": float(i % 2)}
                for i in range(5)
            ]
        }
        resp = client.post(f"/api/models/{mid}/ground-truth", json=labels)
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched"] == 5
        assert data["orphaned"] == 0

    def test_orphaned_labels_reported(self, client):
        mid = _create_model(client)
        labels = {
            "labels": [
                {"predictionId": "ghost-001", "actualValue": 1.0},
                {"predictionId": "ghost-002", "actualValue": 0.0},
            ]
        }
        resp = client.post(f"/api/models/{mid}/ground-truth", json=labels)
        data = resp.json()
        assert data["matched"] == 0
        assert data["orphaned"] == 2


# ── Drift endpoints ──────────────────────────────────────────────────────────

class TestDriftEndpoints:
    def test_drift_empty_returns_list(self, client):
        mid = _create_model(client)
        resp = client.get(f"/api/models/{mid}/drift")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_drift_invalid_window_400(self, client):
        mid = _create_model(client)
        resp = client.get(f"/api/models/{mid}/drift?window=60d")
        assert resp.status_code == 400

    def test_drift_history_requires_feature(self, client):
        mid = _create_model(client)
        resp = client.get(f"/api/models/{mid}/drift/history")
        assert resp.status_code == 422   # missing required query param

    def test_reevaluate_no_baseline(self, client):
        mid = _create_model(client)
        resp = client.post(f"/api/models/{mid}/reevaluate")
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_baseline"

    def test_reevaluate_no_predictions(self, client):
        mid = _create_model(client)
        _upload_baseline(client, mid)
        resp = client.post(f"/api/models/{mid}/reevaluate")
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_predictions"

    def test_reevaluate_with_data_returns_completed(self, client):
        rng = np.random.default_rng(0)
        mid = _create_model(client)
        _upload_baseline(client, mid, n=300)

        # Ingest enough predictions to exceed MIN_SAMPLE_SIZE
        batch = {
            "predictions": [
                {
                    "predictionId": f"r-{i}",
                    "features": {
                        "age": round(float(rng.normal(35, 8)), 2),
                        "income": round(float(rng.normal(50000, 15000)), 2),
                        "credit_score": round(float(rng.normal(700, 50)), 2),
                    },
                    "prediction": float(rng.integers(0, 2)),
                }
                for i in range(300)
            ]
        }
        client.post(f"/api/models/{mid}/predictions", json=batch)

        resp = client.post(f"/api/models/{mid}/reevaluate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert "healthScore" in data
        assert "drifts" in data
        assert len(data["drifts"]) == 3   # age, income, credit_score

    def test_drift_scores_stored_after_reevaluate(self, client):
        rng = np.random.default_rng(1)
        mid = _create_model(client)
        _upload_baseline(client, mid, n=300)
        batch = {
            "predictions": [
                {
                    "predictionId": f"ds-{i}",
                    "features": {
                        "age": round(float(rng.normal(35, 8)), 2),
                        "income": round(float(rng.normal(50000, 15000)), 2),
                        "credit_score": round(float(rng.normal(700, 50)), 2),
                    },
                    "prediction": 0.0,
                }
                for i in range(300)
            ]
        }
        client.post(f"/api/models/{mid}/predictions", json=batch)
        client.post(f"/api/models/{mid}/reevaluate")

        resp = client.get(f"/api/models/{mid}/drift")
        assert resp.status_code == 200
        drifts = resp.json()
        assert len(drifts) == 3
        feature_names = {d["featureName"] for d in drifts}
        assert feature_names == {"age", "income", "credit_score"}
        assert all("jsDistance" in drift for drift in drifts)
        assert all("nLive" in drift for drift in drifts)

    def test_drift_returns_only_latest_score_per_feature(self, client):
        mid = _create_model(client)
        db_session = next(client.app.dependency_overrides[get_db]())
        timestamp = utcnow()
        db_session.add_all([
            DriftScore(
                id="older",
                model_id=mid,
                feature_name="age",
                timestamp=timestamp - timedelta(seconds=1),
                psi_score=0.1,
                js_distance=0.01,
                status="stable",
            ),
            DriftScore(
                id="newer",
                model_id=mid,
                feature_name="age",
                timestamp=timestamp,
                psi_score=0.2,
                js_distance=0.02,
                status="warning",
            ),
        ])
        db_session.commit()

        response = client.get(f"/api/models/{mid}/drift")

        assert response.status_code == 200
        assert len(response.json()) == 1
        assert response.json()[0]["driftScore"] == 0.2


# ── Alert lifecycle ──────────────────────────────────────────────────────────

class TestAlertLifecycle:
    def test_list_alerts_empty(self, client):
        resp = client.get("/api/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_acknowledge_alert(self, client):
        # Create a model and inject a critical drift to generate an alert
        from app.models import Alert as AlertDB, utcnow
        from app.database import SessionLocal

        mid = _create_model(client)

        # Directly insert an alert for testing lifecycle
        from sqlalchemy.orm import Session
        db_session = next(client.app.dependency_overrides[get_db]())
        alert = AlertDB(
            model_id=mid,
            severity="critical",
            category="data_drift",
            title="Test Alert",
            description="Test",
            status="unacknowledged",
            fingerprint=f"{mid}:data_drift:age",
            first_seen_at=utcnow(),
            last_seen_at=utcnow(),
            occurrences=1,
        )
        db_session.add(alert)
        db_session.commit()
        alert_id = alert.id

        resp = client.patch(f"/api/alerts/{alert_id}", json={"status": "investigating"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "investigating"

    def test_resolve_alert(self, client):
        from app.models import Alert as AlertDB, utcnow

        mid = _create_model(client)
        db_session = next(client.app.dependency_overrides[get_db]())
        alert = AlertDB(
            model_id=mid,
            severity="warning",
            category="data_drift",
            title="Resolve Test",
            status="unacknowledged",
            fingerprint=f"{mid}:data_drift:income",
            first_seen_at=utcnow(),
            last_seen_at=utcnow(),
            occurrences=1,
        )
        db_session.add(alert)
        db_session.commit()
        alert_id = alert.id

        resp = client.patch(f"/api/alerts/{alert_id}", json={"status": "resolved"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "resolved"

    def test_filter_active_excludes_resolved(self, client):
        from app.models import Alert as AlertDB, utcnow

        mid = _create_model(client)
        db_session = next(client.app.dependency_overrides[get_db]())
        for i, status in enumerate(["unacknowledged", "resolved", "investigating"]):
            db_session.add(AlertDB(
                model_id=mid, severity="warning", category="data_drift",
                title=f"Alert {i}", status=status,
                fingerprint=f"{mid}:test:{i}",
                first_seen_at=utcnow(), last_seen_at=utcnow(), occurrences=1,
            ))
        db_session.commit()

        resp = client.get("/api/alerts?status=active")
        data = resp.json()
        assert data["total"] == 2   # unacknowledged + investigating
        statuses = {a["status"] for a in data["items"]}
        assert "resolved" not in statuses

    def test_patch_nonexistent_alert_404(self, client):
        resp = client.patch("/api/alerts/no-such-id", json={"status": "resolved"})
        assert resp.status_code == 404


# ── Alert rules ──────────────────────────────────────────────────────────────

class TestAlertRules:
    def test_create_and_list(self, client):
        resp = client.post("/api/alert-rules", json={
            "metric": "psi",
            "thresholdWarning": 0.1,
            "thresholdCritical": 0.25,
            "enabled": True,
        })
        assert resp.status_code == 201
        rule = resp.json()
        assert rule["thresholdWarning"] == 0.1
        assert rule["thresholdCritical"] == 0.25
        assert "threshold_warning" not in rule   # must be camelCase

        resp2 = client.get("/api/alert-rules")
        assert any(r["id"] == rule["id"] for r in resp2.json())

    def test_update_rule(self, client):
        resp = client.post("/api/alert-rules", json={
            "metric": "js_distance",
            "thresholdWarning": 0.05,
            "thresholdCritical": 0.15,
        })
        rule_id = resp.json()["id"]

        patch = client.patch(f"/api/alert-rules/{rule_id}", json={
            "thresholdWarning": 0.08,
            "thresholdCritical": 0.20,
        })
        assert patch.status_code == 200
        updated = patch.json()
        assert updated["thresholdWarning"] == 0.08
        assert updated["thresholdCritical"] == 0.20

    def test_delete_rule(self, client):
        resp = client.post("/api/alert-rules", json={
            "metric": "psi", "thresholdWarning": 0.1, "thresholdCritical": 0.25,
        })
        rule_id = resp.json()["id"]
        del_resp = client.delete(f"/api/alert-rules/{rule_id}")
        assert del_resp.status_code == 204
        rules = client.get("/api/alert-rules").json()
        assert not any(r["id"] == rule_id for r in rules)


# ── Dashboard ────────────────────────────────────────────────────────────────

class TestDashboard:
    def test_summary_empty_db(self, client):
        resp = client.get("/api/dashboard/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "healthAvg" in data
        assert "activeModels" in data
        assert "activeAlerts" in data
        assert "healthOverTime" in data
        assert "featureDriftAlerts" in data
        assert data["activeModels"]["total"] == 0

    def test_summary_with_model(self, client):
        _create_model(client)
        resp = client.get("/api/dashboard/summary")
        data = resp.json()
        assert data["activeModels"]["total"] == 1

    def test_summary_invalid_window(self, client):
        resp = client.get("/api/dashboard/summary?window=60d")
        assert resp.status_code == 400

    def test_public_settings(self, client):
        resp = client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "psiWarning" in data
        assert "psiCritical" in data
        assert "minSampleSize" in data


# ── Reports ──────────────────────────────────────────────────────────────────

class TestReports:
    def test_generate_report(self, client):
        mid = _create_model(client)
        resp = client.post("/api/reports/generate", json={
            "model_id": mid,
            "date_range_start": "2024-01-01T00:00:00",
            "date_range_end": "2024-01-31T23:59:59",
            "report_type": "drift_analysis",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert data["status"] == "generating"

    def test_list_reports(self, client):
        mid = _create_model(client)
        client.post("/api/reports/generate", json={
            "model_id": mid,
            "date_range_start": "2024-01-01T00:00:00",
            "date_range_end": "2024-01-31T23:59:59",
            "report_type": "drift_analysis",
        })
        resp = client.get("/api/reports")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_generate_nonexistent_model_404(self, client):
        resp = client.post("/api/reports/generate", json={
            "model_id": "no-such",
            "date_range_start": "2024-01-01T00:00:00",
            "date_range_end": "2024-01-31T23:59:59",
            "report_type": "drift_analysis",
        })
        assert resp.status_code == 404


# ── Health endpoint ──────────────────────────────────────────────────────────

class TestHealthEndpoints:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
