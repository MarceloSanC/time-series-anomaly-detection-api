from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.dependencies import get_model_service
from app.domain.schemas import DataPoint, DetectorType, ErrorResponse
from app.services.model_service import ModelService

router = APIRouter(tags=["Prediction"])
logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    timestamp: str = Field(..., description="Unix timestamp represented as numeric string.", examples=["1700000100"])
    value: float = Field(..., description="Observed value for anomaly prediction.", examples=[99.0])

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_string(cls, value: str) -> str:
        """Accept only non-empty values coercible to unix-timestamp integer."""
        text = value.strip()
        try:
            int(text)
        except (TypeError, ValueError):
            raise ValueError("timestamp must be a unix timestamp string")
        if text == "":
            raise ValueError("timestamp must be a unix timestamp string")
        return text


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    anomaly: bool = Field(..., description="True when the point exceeds model threshold.", examples=[False])
    model_version: str = Field(..., description="Model version used for prediction.", examples=["v1"])
    detector: DetectorType = Field(
        ...,
        description="Detector type used for prediction.",
        examples=["gaussian", "isolation_forest"],
    )


@router.post(
    "/predict/{series_id}",
    response_model=PredictResponse,
    summary="Predict anomaly for one data point",
    description="Runs prediction for a single timestamp/value point using latest or explicit model version.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid series identifier."},
        404: {"model": ErrorResponse, "description": "Series or requested version not found."},
        422: {"model": ErrorResponse, "description": "Request payload validation error."},
    },
)
def predict_series(
    series_id: str,
    payload: PredictRequest,
    version: str | None = Query(default=None),
    detector: str = Query(
        default="gaussian",
        description="Detector type to use for prediction. Supported values: gaussian, isolation_forest.",
        openapi_examples={
            "gaussian": {"summary": "Default detector", "value": "gaussian"},
            "isolation_forest": {"summary": "Isolation Forest detector", "value": "isolation_forest"},
        },
    ),
    model_service: ModelService = Depends(get_model_service),
) -> PredictResponse:
    """Predict anomaly status for one point using latest or requested version."""
    logger.info("Predict request received", extra={"series_id": series_id, "version": version, "detector": detector})
    timestamp = int(payload.timestamp)

    prediction = model_service.predict(
        series_id=series_id,
        data_point=DataPoint(timestamp=timestamp, value=payload.value),
        version=version,
        detector=detector,
    )
    logger.info(
        "Predict request completed",
        extra={"series_id": series_id, "version": prediction.version, "anomaly": prediction.is_anomaly},
    )
    return PredictResponse(anomaly=prediction.is_anomaly, model_version=prediction.version, detector=prediction.detector)
