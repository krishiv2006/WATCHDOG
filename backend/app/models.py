import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Text, JSON,
    ForeignKey, Boolean, Enum as SAEnum
)
from sqlalchemy.orm import relationship
from app.database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Model(Base):
    __tablename__ = "models"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    algorithm = Column(String(100), nullable=False)
    environment = Column(String(50), nullable=False, default="production")
    status = Column(String(50), nullable=False, default="production")  # production/staging/retired/retraining
    created_at = Column(DateTime, default=datetime.utcnow)
    last_evaluated_at = Column(DateTime, nullable=True)
    health_score = Column(Float, default=100.0)
    training_data_ref = Column(String(255), nullable=True)

    baselines = relationship("ModelBaseline", back_populates="model", cascade="all, delete-orphan")
    predictions = relationship("PredictionLog", back_populates="model", cascade="all, delete-orphan")
    drift_scores = relationship("DriftScore", back_populates="model", cascade="all, delete-orphan")
    performance_snapshots = relationship("PerformanceSnapshot", back_populates="model", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="model", cascade="all, delete-orphan")
    alert_rules = relationship("AlertRule", back_populates="model", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="model", cascade="all, delete-orphan")


class ModelBaseline(Base):
    __tablename__ = "model_baselines"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    feature_name = Column(String(255), nullable=False)
    mean = Column(Float, nullable=True)
    std = Column(Float, nullable=True)
    histogram_counts = Column(JSON, nullable=True)  # list of bucket counts
    histogram_edges = Column(JSON, nullable=True)    # list of bucket edges
    created_at = Column(DateTime, default=datetime.utcnow)

    # Baseline performance metrics (stored once per model, feature_name="__performance__")
    accuracy = Column(Float, nullable=True)
    precision_score = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1 = Column(Float, nullable=True)

    model = relationship("Model", back_populates="baselines")


class PredictionLog(Base):
    __tablename__ = "predictions_log"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    feature_values = Column(JSON, nullable=False)  # dict of feature_name -> value
    predicted_value = Column(Float, nullable=False)
    actual_value = Column(Float, nullable=True)  # ground truth, arrives later
    latency_ms = Column(Float, nullable=True)

    model = relationship("Model", back_populates="predictions")


class DriftScore(Base):
    __tablename__ = "drift_scores"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    feature_name = Column(String(255), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    psi_score = Column(Float, nullable=True)
    js_distance = Column(Float, nullable=True)
    baseline_mean = Column(Float, nullable=True)
    current_mean = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, default="stable")  # stable/warning/critical

    model = relationship("Model", back_populates="drift_scores")


class PerformanceSnapshot(Base):
    __tablename__ = "performance_snapshots"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    accuracy = Column(Float, nullable=True)
    precision_score = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1 = Column(Float, nullable=True)
    sample_size = Column(Integer, default=0)

    model = relationship("Model", back_populates="performance_snapshots")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    severity = Column(String(20), nullable=False)  # critical/warning
    category = Column(String(50), nullable=False)   # data_drift/prediction_drift/performance/latency
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    detected_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(30), nullable=False, default="unacknowledged")  # unacknowledged/investigating/resolved
    resolved_at = Column(DateTime, nullable=True)

    model = relationship("Model", back_populates="alerts")


class AlertRule(Base):
    __tablename__ = "alert_rules"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("models.id"), nullable=True)  # null = global default
    metric = Column(String(100), nullable=False)  # e.g. "psi", "js_distance", "accuracy_delta"
    threshold_warning = Column(Float, nullable=False)
    threshold_critical = Column(Float, nullable=False)
    enabled = Column(Boolean, default=True)

    model = relationship("Model", back_populates="alert_rules")


class Report(Base):
    __tablename__ = "reports"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    date_range_start = Column(DateTime, nullable=False)
    date_range_end = Column(DateTime, nullable=False)
    report_type = Column(String(100), nullable=False)  # drift_analysis/performance/comprehensive
    status = Column(String(20), nullable=False, default="generating")  # ready/generating/failed
    file_path = Column(String(500), nullable=True)
    generated_at = Column(DateTime, nullable=True)
    name = Column(String(255), nullable=True)

    model = relationship("Model", back_populates="reports")
