"""
Watchdog AI — SQLAlchemy ORM models.

Phase 2 redesign: adds task_type, BaselineVersion, enhanced PredictionLog,
enriched DriftScore, PerformanceSnapshot, Alert fingerprinting, and
AlertRule per-feature/cooldown support.  Timezone-aware UTC datetimes.

Backwards-compatible: existing tables are preserved with new nullable columns;
Alembic migrations handle the transition.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Text, JSON,
    ForeignKey, Boolean, Index, UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    """Timezone-aware UTC now — replaces deprecated datetime.utcnow()."""
    return datetime.now(timezone.utc)


# ── Model ──────────────────────────────────────────────────────────────────

class Model(Base):
    __tablename__ = "models"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    algorithm = Column(String(100), nullable=False)
    environment = Column(String(50), nullable=False, default="production")
    # production | staging | retired | retraining
    status = Column(String(50), nullable=False, default="production")
    # binary_classification | multiclass | regression
    task_type = Column(String(50), nullable=True, default="binary_classification")
    # Semantics of the prediction column (e.g. "class_label", "probability", "score")
    prediction_column = Column(String(100), nullable=True)
    # Expected delay between prediction and label availability
    label_delay_hint = Column(String(50), nullable=True)
    # Soft-delete / retirement timestamp
    retired_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    last_evaluated_at = Column(DateTime(timezone=True), nullable=True)
    health_score = Column(Float, nullable=True, default=None)
    health_components = Column(JSON, nullable=True, default=None)
    training_data_ref = Column(String(255), nullable=True)

    # Relationships
    baselines = relationship("ModelBaseline", back_populates="model", cascade="all, delete-orphan")
    baseline_versions = relationship("BaselineVersion", back_populates="model", cascade="all, delete-orphan")
    predictions = relationship("PredictionLog", back_populates="model", cascade="all, delete-orphan")
    drift_scores = relationship("DriftScore", back_populates="model", cascade="all, delete-orphan")
    performance_snapshots = relationship("PerformanceSnapshot", back_populates="model", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="model", cascade="all, delete-orphan")
    alert_rules = relationship("AlertRule", back_populates="model", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="model", cascade="all, delete-orphan")


# ── Legacy baseline (kept for migration transition) ────────────────────────

class ModelBaseline(Base):
    __tablename__ = "model_baselines"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    feature_name = Column(String(255), nullable=False)
    mean = Column(Float, nullable=True)
    std = Column(Float, nullable=True)
    histogram_counts = Column(JSON, nullable=True)
    histogram_edges = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    accuracy = Column(Float, nullable=True)
    precision_score = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1 = Column(Float, nullable=True)

    model = relationship("Model", back_populates="baselines")


# ── BaselineVersion (immutable, one active per model) ──────────────────────

class BaselineVersion(Base):
    """
    Immutable snapshot of training/validation data distribution.

    One active version per model at a time.  Contains per-feature profiles
    as JSON (compact sketches, not raw rows) so baselines of millions of
    rows stay small.
    """
    __tablename__ = "baseline_versions"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    version_number = Column(Integer, nullable=False, default=1)
    is_active = Column(Boolean, nullable=False, default=True)

    # Per-feature profiles (list of dicts):
    # {
    #   "name": str,
    #   "feature_type": "numeric" | "categorical" | "boolean",
    #   "count": int, "null_rate": float,
    #   For numeric: "min", "max", "mean", "std", "quantile_edges" (list),
    #                "histogram_counts" (list), "histogram_edges" (list),
    #                "percentiles" (dict: "25", "50", "75")
    #   For categorical: "category_frequencies" (dict), "cardinality": int,
    #                     "top_k" (list), "other_count": int
    # }
    feature_profiles = Column(JSON, nullable=False, default=list)

    # Number of rows in the source dataset
    sample_size = Column(Integer, nullable=True)

    # Source file tracking
    source_file_name = Column(String(500), nullable=True)
    source_file_hash = Column(String(64), nullable=True)  # SHA-256

    # Baseline performance from labelled validation set (nullable — may not exist)
    performance_metrics = Column(JSON, nullable=True)
    # e.g. {"accuracy": 0.92, "precision": 0.91, "recall": 0.90, "f1": 0.905, "auc": 0.95}
    # or for regression: {"mae": 2.3, "rmse": 3.1, "r2": 0.87}

    # Baseline prediction distribution
    prediction_distribution = Column(JSON, nullable=True)
    # e.g. {"class_balance": {"0": 0.6, "1": 0.4}} or {"score_histogram": {...}}

    # Number of quantile buckets for PSI (configurable per baseline)
    n_psi_buckets = Column(Integer, nullable=False, default=10)

    created_at = Column(DateTime(timezone=True), default=utcnow)

    model = relationship("Model", back_populates="baseline_versions")
    drift_scores = relationship("DriftScore", back_populates="baseline_version")

    __table_args__ = (
        Index("ix_baseline_model_active", "model_id", "is_active"),
    )


# ── PredictionLog ──────────────────────────────────────────────────────────

class PredictionLog(Base):
    __tablename__ = "predictions_log"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    # Client-supplied or generated; unique per model for idempotency + label join
    prediction_id = Column(String(255), nullable=True, index=True)
    model_version = Column(String(50), nullable=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, index=True)
    feature_values = Column(JSON, nullable=False)
    predicted_value = Column(Float, nullable=True)
    # Score/probability (nullable; only for classification with probabilities)
    score = Column(Float, nullable=True)
    actual_value = Column(Float, nullable=True)
    labeled_at = Column(DateTime(timezone=True), nullable=True)
    latency_ms = Column(Float, nullable=True)

    model = relationship("Model", back_populates="predictions")

    __table_args__ = (
        Index("ix_pred_model_timestamp", "model_id", "timestamp"),
        Index("ix_pred_model_predid", "model_id", "prediction_id"),
        UniqueConstraint("model_id", "prediction_id", name="uq_pred_model_prediction_id"),
    )


class AnalyzeCSVUpload(Base):
    """Idempotency ledger and cached result for analyze-csv uploads."""

    __tablename__ = "analyze_csv_uploads"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    file_hash = Column(String(64), nullable=False)
    result = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        UniqueConstraint("model_id", "file_hash", name="uq_analyze_csv_model_hash"),
        Index("ix_analyze_csv_model", "model_id"),
    )


# ── DriftScore ─────────────────────────────────────────────────────────────

class DriftScore(Base):
    __tablename__ = "drift_scores"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    feature_name = Column(String(255), nullable=False)
    timestamp = Column(DateTime(timezone=True), default=utcnow, index=True)

    # Core metrics
    psi_score = Column(Float, nullable=True)
    js_distance = Column(Float, nullable=True)
    ks_statistic = Column(Float, nullable=True)
    ks_pvalue = Column(Float, nullable=True)

    # Distribution summary
    baseline_mean = Column(Float, nullable=True)
    current_mean = Column(Float, nullable=True)
    baseline_std = Column(Float, nullable=True)
    current_std = Column(Float, nullable=True)

    # Data quality
    null_rate_baseline = Column(Float, nullable=True)
    null_rate_live = Column(Float, nullable=True)
    out_of_range_rate = Column(Float, nullable=True)

    # Sample sizes
    n_live = Column(Integer, nullable=True)
    n_baseline = Column(Integer, nullable=True)

    # Decision output
    # stable | warning | critical | insufficient_data
    status = Column(String(20), nullable=False, default="stable")
    confidence = Column(String(10), nullable=True)  # high | medium | low
    reason = Column(Text, nullable=True)

    # Link to which baseline version was used
    baseline_version_id = Column(String, ForeignKey("baseline_versions.id"), nullable=True)

    model = relationship("Model", back_populates="drift_scores")
    baseline_version = relationship("BaselineVersion", back_populates="drift_scores")

    __table_args__ = (
        Index("ix_drift_model_feature_ts", "model_id", "feature_name", "timestamp"),
    )


# ── PerformanceSnapshot ────────────────────────────────────────────────────

class PerformanceSnapshot(Base):
    __tablename__ = "performance_snapshots"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), default=utcnow, index=True)

    # Classification metrics
    accuracy = Column(Float, nullable=True)
    precision_score = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1 = Column(Float, nullable=True)
    auc = Column(Float, nullable=True)

    # Regression metrics
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    r_squared = Column(Float, nullable=True)

    # Metadata
    sample_size = Column(Integer, default=0)
    n_labeled = Column(Integer, nullable=True)
    label_coverage = Column(Float, nullable=True)  # fraction of predictions with labels

    # Status
    # healthy | degraded | awaiting_labels
    perf_status = Column(String(30), nullable=True)

    model = relationship("Model", back_populates="performance_snapshots")


# ── Alert ──────────────────────────────────────────────────────────────────

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    severity = Column(String(20), nullable=False)  # critical | warning
    category = Column(String(50), nullable=False)  # data_drift | prediction_drift | performance | latency | data_quality
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    detected_at = Column(DateTime(timezone=True), default=utcnow)
    status = Column(String(30), nullable=False, default="unacknowledged")
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    # Deduplication
    fingerprint = Column(String(255), nullable=True, index=True)
    first_seen_at = Column(DateTime(timezone=True), nullable=True)
    last_seen_at = Column(DateTime(timezone=True), nullable=True)
    occurrences = Column(Integer, nullable=True, default=1)
    auto_resolved = Column(Boolean, nullable=True, default=False)

    # Link to triggering rule and metric
    alert_rule_id = Column(String, ForeignKey("alert_rules.id"), nullable=True)
    metric_value = Column(Float, nullable=True)

    model = relationship("Model", back_populates="alerts")
    alert_rule = relationship("AlertRule", back_populates="triggered_alerts")


# ── AlertRule ──────────────────────────────────────────────────────────────

class AlertRule(Base):
    __tablename__ = "alert_rules"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("models.id"), nullable=True)  # null = global default
    metric = Column(String(100), nullable=False)
    threshold_warning = Column(Float, nullable=False)
    threshold_critical = Column(Float, nullable=False)
    enabled = Column(Boolean, default=True)

    # Per-feature targeting (null = applies to all features)
    feature_name = Column(String(255), nullable=True)
    # Comparison operator: "gt" (greater than), "lt" (less than)
    comparison = Column(String(10), nullable=True, default="gt")
    # Minimum live samples before this rule fires
    min_samples = Column(Integer, nullable=True)
    # Consecutive flagged windows required before alert
    consecutive_windows = Column(Integer, nullable=True, default=2)
    # Cooldown between repeated alerts (minutes)
    cooldown_minutes = Column(Integer, nullable=True, default=60)
    # Notification channels: JSON list e.g. ["email", "webhook", "slack"]
    channels = Column(JSON, nullable=True)

    model = relationship("Model", back_populates="alert_rules")
    triggered_alerts = relationship("Alert", back_populates="alert_rule")


# ── Report ─────────────────────────────────────────────────────────────────

class Report(Base):
    __tablename__ = "reports"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    date_range_start = Column(DateTime(timezone=True), nullable=False)
    date_range_end = Column(DateTime(timezone=True), nullable=False)
    report_type = Column(String(100), nullable=False)
    format = Column(String(20), nullable=True, default="csv")  # csv | pdf | json
    status = Column(String(20), nullable=False, default="generating")
    file_path = Column(String(500), nullable=True)
    generated_at = Column(DateTime(timezone=True), nullable=True)
    name = Column(String(255), nullable=True)

    model = relationship("Model", back_populates="reports")
