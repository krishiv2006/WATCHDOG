from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Model schemas ──────────────────────────────────
class ModelBase(BaseModel):
    name: str
    version: str
    algorithm: str
    environment: str = "production"
    status: str = "production"
    training_data_ref: Optional[str] = None


class ModelCreate(ModelBase):
    pass


class ModelOut(ModelBase):
    id: str
    health_score: float
    created_at: datetime
    last_evaluated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ModelSummary(BaseModel):
    id: str
    name: str
    version: str
    algorithm: str
    environment: str
    status: str
    predictions_count: int = 0
    health_score: float
    drift_status: str = "Low"
    last_updated: str = ""

    class Config:
        from_attributes = True


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
    feature_drift_alerts: list[FeatureDriftAlert] = Field(default_factory=list, alias="featureDriftAlerts")

    class Config:
        populate_by_name = True


# ── Drift schemas ──────────────────────────────────
class DriftScoreOut(BaseModel):
    feature_name: str = Field(alias="featureName")
    baseline_mean: float = Field(alias="baselineMean")
    current_mean: float = Field(alias="currentMean")
    drift_score: float = Field(alias="driftScore")
    js_distance: Optional[float] = Field(None, alias="jsDistance")
    status: str

    class Config:
        populate_by_name = True
        from_attributes = True


# ── Performance schemas ────────────────────────────
class PerformanceMetric(BaseModel):
    label: str
    value: float
    baseline: float
    delta: float


class PerformanceTimePoint(BaseModel):
    time: str
    current: float
    base: float


class PerformanceOut(BaseModel):
    metrics: list[PerformanceMetric]
    time_series: list[PerformanceTimePoint] = Field(alias="timeSeries")

    class Config:
        populate_by_name = True


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

    class Config:
        populate_by_name = True
        from_attributes = True


class AlertUpdate(BaseModel):
    status: str  # unacknowledged/investigating/resolved


class PaginatedAlerts(BaseModel):
    items: list[AlertOut]
    total: int
    page: int
    per_page: int = Field(alias="perPage")
    total_pages: int = Field(alias="totalPages")

    class Config:
        populate_by_name = True


# ── Alert Rule schemas ─────────────────────────────
class AlertRuleBase(BaseModel):
    model_id: Optional[str] = None
    metric: str
    threshold_warning: float
    threshold_critical: float
    enabled: bool = True


class AlertRuleCreate(AlertRuleBase):
    pass


class AlertRuleUpdate(BaseModel):
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    enabled: Optional[bool] = None


class AlertRuleOut(AlertRuleBase):
    id: str

    class Config:
        from_attributes = True


# ── Report schemas ─────────────────────────────────
class ReportCreate(BaseModel):
    model_id: str
    date_range_start: str
    date_range_end: str
    report_type: str = "drift_analysis"


class ReportOut(BaseModel):
    id: str
    name: Optional[str] = None
    model_id: str
    model_name: str = ""
    date_range: str = ""
    status: str
    generated_at: Optional[str] = None

    class Config:
        from_attributes = True


# ── WebSocket event ────────────────────────────────
class WSAlertEvent(BaseModel):
    event: str  # "new_alert" or "alert_updated"
    alert: AlertOut
