from datetime import timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import Model, PredictionLog, utcnow
from app.models import BaselineVersion
from app.services.quality import feature_null_spikes, schema_warnings, volume_anomaly


def _session() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session = Session(engine)
    session.info["engine"] = engine
    return session


def test_back_to_back_uploads_without_history_do_not_trigger_volume_alert():
    db = _session()
    model = Model(name="volume", version="v1", algorithm="test")
    db.add(model)
    db.flush()
    now = utcnow()
    for batch in range(5):
        for row in range(20):
            db.add(PredictionLog(
                model_id=model.id,
                prediction_id=f"{batch}-{row}",
                timestamp=now,
                feature_values={"x": 1},
                predicted_value=0,
            ))
    db.commit()

    assert volume_anomaly(db, model.id) is None
    db.close()


def test_volume_anomaly_detects_spike_after_history():
    db = _session()
    model = Model(name="volume", version="v1", algorithm="test")
    db.add(model)
    db.flush()
    now = utcnow()
    for day in range(1, 15):
        timestamp = now - timedelta(days=day, hours=12)
        for row in range(10):
            db.add(PredictionLog(
                model_id=model.id,
                prediction_id=f"normal-{day}-{row}",
                timestamp=timestamp,
                feature_values={"x": 1},
                predicted_value=0,
            ))
    for row in range(100):
        db.add(PredictionLog(
            model_id=model.id,
            prediction_id=f"spike-{row}",
            timestamp=now,
            feature_values={"x": 1},
            predicted_value=0,
        ))
    db.commit()

    result = volume_anomaly(db, model.id)
    assert result is not None
    assert result["current"] == 100
    assert result["expected"] == 10.0
    db.close()


def test_target_and_unknown_columns_are_excluded_from_quality_alerts():
    baseline = BaselineVersion(
        model_id="model",
        feature_profiles=[
            {"name": "feature", "null_rate": 0},
            {"name": "Outcome", "null_rate": 0},
        ],
        prediction_distribution={"target_column": "Outcome"},
    )
    prediction = PredictionLog(
        feature_values={"feature": 1, "Outcome": None, "unknown": None},
    )

    assert feature_null_spikes([prediction], baseline) == []
    assert schema_warnings({"feature": 1, "Outcome": None, "unknown": None}, baseline) == []


def test_missing_all_values_without_target_reports_schema_not_null_spike():
    baseline = BaselineVersion(
        model_id="model",
        feature_profiles=[{"name": "feature", "null_rate": 0}],
    )
    prediction = PredictionLog(feature_values={})

    assert feature_null_spikes([prediction], baseline) == []
    assert schema_warnings({}, baseline) == ["missing features: ['feature']"]
