"""
Watchdog AI — Pydantic request/response schemas.

Phase 2: adds baseline-version schemas, enhanced DriftScore output,
prediction ingestion schemas, ground-truth ingestion, and enriched
alert/performance schemas.

Convention: all output schemas emit camelCase via Field(alias=...) and
``model_config = ConfigDict(populate_by_name=True, from_attributes=True)``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict


# ── Model schemas ──────────────────────────────────


class ModelCreate(BaseModel):
    name: str
    version: str
    algorithm: str
    environment: str = "production"
    status: str = "production"
    task_type: str = Field("binary_classification", alias="taskType")
    prediction_column: Optional[str] = Field(None, alias="predictionColumn")
    label_delay_hint: Optional[str] = Field(None, alias="labelDelayHint")
    training_data_ref: Optional[str] = Field(None, alias="trainingDataRef")

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


class ModelOut(BaseModel):
    id: str
    name: str
    version: str
    algorithm: str
    environment: str
    status: str
    task_type: Optional[str] = Field(None, alias="taskType")
    health_score: float = Field(alias="healthScore")
    created_at: Optional[datetime] = Field(None, alias="createdAt")
    last_evaluated_at: Optional[datetime] = Field(None, alias="lastEvaluatedAt")
    predictions_count: int = Field(0, alias="predictionsCount")
    drift_status: str = Field("Low", alias="driftStatus")
    last_updated: str = Field("", alias="lastUpdated")
    training_data_ref: Optional[str] = Field(None, alias="trainingDataRef")

    model_config = ConfigDict(populate_by_name=True, from_attributes=True, protected_namespaces=())


class ModelSummary(BaseModel):
    id: str
    name: str
    version: str
    algorithm: str
    environment: str
    status: str
    predictions_count: int = Field(0, alias="predictionsCount")
    health_score: float = Field(alias="healthScore")
    drift_status: str = Field("Low", alias="driftStatus")
    last_updated: str = Field("", alias="lastUpdated")

    model_config = ConfigDict(populate_by_name=True, from_attributes=True, protected_namespaces=())


# ── Dashboard schemas ──────────────────────────────


class ActiveModelsInfo(BaseModel):
    total: int
    prod: int
    staging: int


class ActiveAlertsInfo(BaseModel):
    total: int
    critical: int
    warning: int


class HealthTimePoint(BaseModel):
    time: str
    value: float


class FeatureDriftAlert(BaseModel):
    feature: str
    score: str
    status: str


class DashboardSummary(BaseModel):
    health_avg: float = Field(alias="healthAvg")
    active_models: ActiveModelsInfo = Field(alias="activeModels")
    predictions_total: int = Field(alias="predictionsTotal")
    active_alerts: ActiveAlertsInfo = Field(alias="activeAlerts")
    health_over_time: list[HealthTimePoint] = Field(alias="healthOverTime")
    feature_drift_alerts: list[FeatureDriftAlert] = Field(
        default_factory=list, alias="featureDriftAlerts"
    )

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


# ── Baseline schemas ──────────────────────────────

class BaselineVersionOut(BaseModel):
    id: str
    model_id: str = Field(alias="modelId")
    version_number: int = Field(alias="versionNumber")
    is_active: bool = Field(alias="isActive")
    sample_size: Optional[int] = Field(None, alias="sampleSize")
    source_file_name: Optional[str] = Field(None, alias="sourceFileName")
    n_psi_buckets: int = Field(alias="nPsiBuckets")
    performance_metrics: Optional[dict] = Field(None, alias="performanceMetrics")
    feature_count: int = Field(0, alias="featureCount")
    created_at: Optional[str] = Field(None, alias="createdAt")

    model_config = ConfigDict(populate_by_name=True, from_attributes=True, protected_namespaces=())


# ── Drift schemas ──────────────────────────────────


class DriftScoreOut(BaseModel):
    feature_name: str = Field(alias="featureName")
    baseline_mean: Optional[float] = Field(None, alias="baselineMean")
    current_mean: Optional[float] = Field(None, alias="currentMean")
    drift_score: Optional[float] = Field(None, alias="driftScore")
    js_distance: Optional[float] = Field(None, alias="jsDistance")
    ks_statistic: Optional[float] = Field(None, alias="ksStatistic")
    ks_pvalue: Optional[float] = Field(None, alias="ksPvalue")
    status: str
    confidence: Optional[str] = None
    reason: Optional[str] = None
    n_live: Optional[int] = Field(None, alias="nLive")
    n_baseline: Optional[int] = Field(None, alias="nBaseline")
    null_rate_baseline: Optional[float] = Field(None, alias="nullRateBaseline")
    null_rate_live: Optional[float] = Field(None, alias="nullRateLive")
    out_of_range_rate: Optional[float] = Field(None, alias="outOfRangeRate")
    timestamp: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, from_attributes=True, protected_namespaces=())


class DriftDecisionOut(BaseModel):
    """Detailed drift decision result for a single feature."""

    feature_name: str = Field(alias="featureName")
    status: str  # insufficient_data | stable | warning | critical
    confidence: str  # high | medium | low
    reason: str
    psi: Optional[float] = None
    js_distance: Optional[float] = Field(None, alias="jsDistance")
    ks_statistic: Optional[float] = Field(None, alias="ksStatistic")
    ks_pvalue: Optional[float] = Field(None, alias="ksPvalue")
    noise_floor: Optional[float] = Field(None, alias="noiseFloor")
    n_live: int = Field(alias="nLive")
    n_baseline: Optional[int] = Field(None, alias="nBaseline")
    null_rate_baseline: Optional[float] = Field(None, alias="nullRateBaseline")
    null_rate_live: Optional[float] = Field(None, alias="nullRateLive")

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


# ── Performance schemas ────────────────────────────


class PerformanceMetric(BaseModel):
    label: str
    value: Optional[float] = None
    baseline: Optional[float] = None
    delta: Optional[float] = None
    reason: Optional[str] = None


class PerformanceTimePoint(BaseModel):
    time: str
    current: Optional[float] = None
    base: Optional[float] = None


class PerformanceOut(BaseModel):
    metrics: list[PerformanceMetric]
    time_series: list[PerformanceTimePoint] = Field(alias="timeSeries")
    has_labels: bool = Field(False, alias="hasLabels")
    sample_size: int = Field(0, alias="sampleSize")
    label_coverage: Optional[float] = Field(None, alias="labelCoverage")
    task_type: Optional[str] = Field(None, alias="taskType")

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


# ── Prediction ingestion schemas ───────────────────


class PredictionInput(BaseModel):
    """Single prediction for ingestion."""

    prediction_id: Optional[str] = Field(None, alias="predictionId")
    timestamp: Optional[str] = None
    features: dict[str, Any]
    prediction: float
    score: Optional[float] = None
    latency_ms: Optional[float] = Field(None, alias="latencyMs")
    model_version: Optional[str] = Field(None, alias="modelVersion")

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


class PredictionBatch(BaseModel):
    """Batch of predictions."""

    predictions: list[PredictionInput]


class PredictionIngestionResult(BaseModel):
    ingested: int
    duplicates: int
    warnings: list[str] = []


# ── Ground truth schemas ───────────────────────────


class GroundTruthInput(BaseModel):
    prediction_id: str = Field(alias="predictionId")
    actual_value: float = Field(alias="actualValue")
    labeled_at: Optional[str] = Field(None, alias="labeledAt")

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


class GroundTruthBatch(BaseModel):
    labels: list[GroundTruthInput]


class GroundTruthResult(BaseModel):
    matched: int
    orphaned: int
    warnings: list[str] = []


# ── Alert schemas ──────────────────────────────────


class AlertOut(BaseModel):
    id: str
    model_id: str = Field(alias="modelId")
    model_name: str = Field(alias="modelName")
    severity: str
    category: str
    title: str
    description: Optional[str] = None
    detected_at: str = Field(alias="detectedAt")
    status: str
    fingerprint: Optional[str] = None
    occurrences: Optional[int] = None
    first_seen_at: Optional[str] = Field(None, alias="firstSeenAt")
    last_seen_at: Optional[str] = Field(None, alias="lastSeenAt")
    auto_resolved: Optional[bool] = Field(None, alias="autoResolved")
    metric_value: Optional[float] = Field(None, alias="metricValue")

    model_config = ConfigDict(populate_by_name=True, from_attributes=True, protected_namespaces=())


class AlertUpdate(BaseModel):
    status: str  # unacknowledged | investigating | resolved


class PaginatedAlerts(BaseModel):
    items: list[AlertOut]
    total: int
    page: int
    per_page: int = Field(alias="perPage")
    total_pages: int = Field(alias="totalPages")

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


# ── Alert Rule schemas ─────────────────────────────


class AlertRuleBase(BaseModel):
    model_id: Optional[str] = Field(None, alias="modelId")
    metric: str
    threshold_warning: float = Field(alias="thresholdWarning")
    threshold_critical: float = Field(alias="thresholdCritical")
    enabled: bool = True
    feature_name: Optional[str] = Field(None, alias="featureName")
    comparison: Optional[str] = "gt"
    min_samples: Optional[int] = Field(None, alias="minSamples")
    consecutive_windows: Optional[int] = Field(2, alias="consecutiveWindows")
    cooldown_minutes: Optional[int] = Field(60, alias="cooldownMinutes")
    channels: Optional[list[str]] = None

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


class AlertRuleCreate(BaseModel):
    """Input schema — accepts both camelCase and snake_case for flexibility."""

    model_id: Optional[str] = Field(None, alias="modelId")
    metric: str
    threshold_warning: float = Field(alias="thresholdWarning")
    threshold_critical: float = Field(alias="thresholdCritical")
    enabled: bool = True
    feature_name: Optional[str] = Field(None, alias="featureName")
    comparison: Optional[str] = None
    min_samples: Optional[int] = Field(None, alias="minSamples")
    consecutive_windows: Optional[int] = Field(None, alias="consecutiveWindows")
    cooldown_minutes: Optional[int] = Field(None, alias="cooldownMinutes")
    channels: Optional[list[str]] = None

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


class AlertRuleUpdate(BaseModel):
    threshold_warning: Optional[float] = Field(None, alias="thresholdWarning")
    threshold_critical: Optional[float] = Field(None, alias="thresholdCritical")
    enabled: Optional[bool] = None
    feature_name: Optional[str] = Field(None, alias="featureName")
    comparison: Optional[str] = None
    min_samples: Optional[int] = Field(None, alias="minSamples")
    consecutive_windows: Optional[int] = Field(None, alias="consecutiveWindows")
    cooldown_minutes: Optional[int] = Field(None, alias="cooldownMinutes")
    channels: Optional[list[str]] = None

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


class AlertRuleOut(BaseModel):
    """Output schema — always emits camelCase to match frontend contract."""

    id: str
    model_id: Optional[str] = Field(None, alias="modelId")
    metric: str
    threshold_warning: float = Field(alias="thresholdWarning")
    threshold_critical: float = Field(alias="thresholdCritical")
    enabled: bool
    feature_name: Optional[str] = Field(None, alias="featureName")
    comparison: Optional[str] = None
    min_samples: Optional[int] = Field(None, alias="minSamples")
    consecutive_windows: Optional[int] = Field(None, alias="consecutiveWindows")
    cooldown_minutes: Optional[int] = Field(None, alias="cooldownMinutes")
    channels: Optional[list[str]] = None

    model_config = ConfigDict(from_attributes=True, populate_by_name=True, protected_namespaces=())


# ── Report schemas ─────────────────────────────────


class ReportCreate(BaseModel):
    model_id: str
    date_range_start: str
    date_range_end: str
    report_type: str = "drift_analysis"
    format: str = "csv"

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


class ReportOut(BaseModel):
    id: str
    name: Optional[str] = None
    model_id: str = Field(alias="modelId")
    model_name: str = Field("", alias="modelName")
    date_range: str = Field("", alias="dateRange")
    status: str
    generated_at: Optional[str] = Field(None, alias="generatedAt")
    format: Optional[str] = None

    model_config = ConfigDict(from_attributes=True, populate_by_name=True, protected_namespaces=())


# ── Health score schema ────────────────────────────


class HealthScoreOut(BaseModel):
    score: float
    components: dict[str, Any]


# ── WebSocket event ────────────────────────────────


class WSAlertEvent(BaseModel):
    event: str  # "new_alert" or "alert_updated"
    alert: AlertOut
