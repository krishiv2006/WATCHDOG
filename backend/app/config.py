from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./watchdog.db"
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    DRIFT_CHECK_INTERVAL_MINUTES: int = 15
    LOG_LEVEL: str = "INFO"

    # Health score weights
    HEALTH_WEIGHT_DRIFT: float = 0.4
    HEALTH_WEIGHT_PERFORMANCE: float = 0.4
    HEALTH_WEIGHT_ALERTS: float = 0.2
    HEALTH_ALERT_PENALTY: float = 5.0  # points subtracted per open critical alert

    # PSI thresholds (default; per-model AlertRule rows override these)
    PSI_THRESHOLD_WARNING: float = 0.1
    PSI_THRESHOLD_CRITICAL: float = 0.25

    # Jensen-Shannon distance thresholds (used in drift_decision cross-check)
    JS_THRESHOLD_WARNING: float = 0.05
    JS_THRESHOLD_CRITICAL: float = 0.15

    # Minimum live-sample size below which status is "insufficient_data"
    # rather than stable/warning/critical (prevents false alarms on tiny batches)
    MIN_SAMPLE_SIZE: int = 250

    # Number of consecutive flagged windows required before creating a critical alert
    DRIFT_CONSECUTIVE_WINDOWS: int = 2

    # Maximum rows per batch prediction ingestion call
    MAX_BATCH_PREDICTIONS: int = 5000

    # Live-window length used when scoring a model against its baseline
    EVAL_WINDOW_HOURS: int = 24

    # Minimum labelled predictions required before reporting performance
    MIN_LABELS: int = 30

    # Upload / schema limits
    MAX_UPLOAD_BYTES: int = 100 * 1024 * 1024
    MAX_FEATURES: int = 500

    # Report files live outside the repo working tree by default
    REPORTS_DIR: str = "./reports_output"

    # Optional shared ingest key. Empty = open (local/dev).
    INGEST_API_KEY: str = ""

    ENABLE_SCHEDULER: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
