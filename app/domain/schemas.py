from typing import Optional, Sequence

from pydantic import BaseModel, Field


class DataPoint(BaseModel):
    timestamp: int = Field(..., description="Unix timestamp")
    value: float = Field(..., description="Measured value")


class TimeSeries(BaseModel):
    data: Sequence[DataPoint] = Field(..., description="Ordered list of data points")


class TrainResponse(BaseModel):
    series_id: str
    version: str
    n_samples: int
    mean: float
    std: float
    training_duration_ms: float
    trained_at: str


class PredictionResponse(BaseModel):
    series_id: str
    version: str
    is_anomaly: bool
    value: float
    timestamp: int
    mean: float
    upper_bound: float


class ModelInfo(BaseModel):
    series_id: str
    latest_version: str
    versions: list[str]
    trained_at: str
    n_samples: int


class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: Optional[str] = None
    timestamp: str
