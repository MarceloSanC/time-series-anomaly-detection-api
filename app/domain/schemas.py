from typing import Optional, Sequence

from pydantic import BaseModel, Field


class DataPoint(BaseModel):
    """Single timestamped measurement in a time series."""

    timestamp: int = Field(..., description="Unix timestamp")
    value: float = Field(..., description="Measured value")


class TimeSeries(BaseModel):
    """Ordered collection of data points used for model training."""

    data: Sequence[DataPoint] = Field(..., description="Ordered list of data points")


class TrainResponse(BaseModel):
    """Internal response payload returned after training a model version."""

    series_id: str
    version: str
    n_samples: int
    mean: float
    std: float
    training_duration_ms: float
    trained_at: str


class PredictionResponse(BaseModel):
    """Internal response payload returned for a prediction request."""

    series_id: str
    version: str
    is_anomaly: bool
    value: float
    timestamp: int
    mean: float
    upper_bound: float


class ModelInfo(BaseModel):
    """Summary metadata for a trained time-series model lineage."""

    series_id: str
    latest_version: str
    versions: list[str]
    trained_at: str
    n_samples: int


class ErrorResponse(BaseModel):
    """Standardized error payload used by API error handlers."""

    error: str
    message: str
    detail: Optional[str] = None
    timestamp: str
