from typing import Any, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

DetectorType = Literal["gaussian", "isolation_forest"]


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
    detector: DetectorType = "gaussian"
    model_params: Optional[dict[str, Any]] = None
    training_duration_ms: float
    trained_at: str


class PredictionResponse(BaseModel):
    """Internal response payload returned for a prediction request."""

    series_id: str
    version: str
    is_anomaly: bool
    value: float
    timestamp: int
    detector: DetectorType = "gaussian"
    model_params: Optional[dict[str, Any]] = None


class ModelInfo(BaseModel):
    """Summary metadata for a trained time-series model lineage."""

    series_id: str
    latest_version: str
    versions: list[str]
    trained_at: str
    n_samples: int


class DataQualityReport(BaseModel):
    """Derived training-data quality indicators for a model lineage."""

    n_samples: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min_value: float
    max_value: float
    time_span_seconds: int
    points_per_second: float


class ModelSummary(BaseModel):
    """Compact per-series summary used by the `/models` list endpoint."""

    series_id: str
    latest_version: str
    n_samples: int
    trained_at: str


class ModelDetail(BaseModel):
    """Detailed per-series payload used by the `/models/{series_id}` endpoint."""

    series_id: str
    latest_version: str
    versions: list[str]
    trained_at: str
    n_samples: int
    data_quality: DataQualityReport


class DataRange(BaseModel):
    """Timestamp range summary stored in metadata."""

    min_timestamp: int
    max_timestamp: int


class MetadataTrainingPoint(BaseModel):
    """Training sample representation persisted in model metadata."""

    timestamp: int
    value: float


class ModelVersionMetadata(BaseModel):
    """Version-level metadata response for model introspection endpoints."""

    model_config = ConfigDict(protected_namespaces=())

    version: str
    detector: DetectorType = "gaussian"
    model_params: Optional[dict[str, Any]] = None
    n_samples: int
    trained_at: str
    training_duration_ms: float
    data_range: DataRange
    training_data: Optional[list[MetadataTrainingPoint]] = None


class ErrorResponse(BaseModel):
    """Standardized error payload used by API error handlers."""

    error: str
    message: str
    detail: Optional[str] = None
    timestamp: str
