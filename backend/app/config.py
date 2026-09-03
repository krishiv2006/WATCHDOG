from pydantic_settings import BaseSettings
from functools import lru_cache


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

    # PSI thresholds
    PSI_THRESHOLD_WARNING: float = 0.1
    PSI_THRESHOLD_CRITICAL: float = 0.25

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
