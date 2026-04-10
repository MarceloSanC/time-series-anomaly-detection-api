from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="Time Series Anomaly Detection API")
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    storage_path: Path = Field(default=Path("storage"))

    min_data_points: int = Field(default=30)
    std_threshold: float = Field(default=1e-10)

    latency_window_size: int = Field(default=1000)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
