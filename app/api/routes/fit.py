from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, model_validator

from app.dependencies import get_model_service
from app.domain.schemas import DataPoint, DetectorType, ErrorResponse, TimeSeries
from app.services.model_service import ModelService

router = APIRouter(tags=["Training"])
logger = logging.getLogger(__name__)


class FitRequest(BaseModel):
    timestamps: list[int] = Field(
        ...,
        description="Unix timestamps for each observed value.",
        examples=[
            [
                1700000001, 1700000002, 1700000003, 1700000004, 1700000005,
                1700000006, 1700000007, 1700000008, 1700000009, 1700000010,
                1700000011, 1700000012, 1700000013, 1700000014, 1700000015,
                1700000016, 1700000017, 1700000018, 1700000019, 1700000020,
                1700000021, 1700000022, 1700000023, 1700000024, 1700000025,
                1700000026, 1700000027, 1700000028, 1700000029, 1700000030,
            ]
        ],
    )
    values: list[float] = Field(
        ...,
        description="Observed values aligned with timestamps.",
        examples=[
            [
                10.0, 10.3, 10.6, 10.9, 11.2,
                11.5, 11.8, 12.0, 12.3, 12.6,
                12.9, 13.2, 13.5, 13.8, 14.0,
                14.3, 14.6, 14.9, 15.2, 15.5,
                15.8, 16.1, 16.4, 16.7, 17.0,
                17.3, 17.6, 17.9, 18.2, 18.5,
            ]
        ],
    )

    @model_validator(mode="after")
    def validate_lengths(self) -> "FitRequest":
        """Validate non-empty aligned timestamp/value arrays."""
        if not self.timestamps:
            raise ValueError("timestamps and values cannot be empty")
        if len(self.timestamps) != len(self.values):
            raise ValueError("timestamps and values must have the same length")
        return self


class FitResponse(BaseModel):
    series_id: str = Field(..., description="Series identifier used for training.", examples=["sensor_XYZ"])
    detector: DetectorType = Field(
        ...,
        description="Detector type used for training.",
        examples=["gaussian", "isolation_forest"],
    )
    version: str = Field(..., description="Persisted model version created by this training call.", examples=["v1"])
    points_used: int = Field(..., description="Number of data points consumed during training.", examples=[120])


@router.post(
    "/fit/{series_id}",
    response_model=FitResponse,
    summary="Train model for one series",
    description="Trains a new model version for the provided `series_id` using aligned timestamp/value arrays.",
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Domain validation error (including data-quality checks).",
            "content": {
                "application/json": {
                    "examples": {
                        "insufficient_data": {
                            "summary": "Insufficient points for training",
                            "value": {
                                "error": "INSUFFICIENT_DATA",
                                "message": "Time series must contain at least 30 points",
                                "detail": None,
                                "timestamp": "2026-04-15T16:45:00Z",
                            },
                        },
                        "flat_line_detected": {
                            "summary": "Trailing flat-line detected",
                            "value": {
                                "error": "FLAT_LINE_DETECTED",
                                "message": "Last 10 values are identical - possible sensor disconnect",
                                "detail": None,
                                "timestamp": "2026-04-15T16:45:00Z",
                            },
                        },
                    }
                }
            },
        },
        422: {
            "model": ErrorResponse,
            "description": "Request payload or detector validation error.",
            "content": {
                "application/json": {
                    "examples": {
                        "unsupported_detector": {
                            "summary": "Unsupported detector query value",
                            "value": {
                                "error": "UNSUPPORTED_DETECTOR",
                                "message": "Detector 'random_forest' is not supported",
                                "detail": None,
                                "timestamp": "2026-04-15T16:45:00Z",
                            },
                        },
                        "payload_validation_error": {
                            "summary": "Invalid request body",
                            "value": {
                                "error": "VALIDATION_ERROR",
                                "message": "Request payload validation failed",
                                "detail": (
                                    "[{'type': 'value_error', 'loc': ['body', 'timestamps'],"
                                    " 'msg': 'timestamps and values cannot be empty'}]"
                                ),
                                "timestamp": "2026-04-15T16:45:00Z",
                            },
                        },
                    }
                }
            },
        },
    },
)
def fit_series(
    series_id: str,
    payload: FitRequest,
    detector: str = Query(
        default="gaussian",
        description="Detector type to train. Supported values: gaussian, isolation_forest.",
        openapi_examples={
            "gaussian": {"summary": "Default detector", "value": "gaussian"},
            "isolation_forest": {"summary": "Isolation Forest detector", "value": "isolation_forest"},
        },
    ),
    model_service: ModelService = Depends(get_model_service),
) -> FitResponse:
    """Train a model for the given `series_id` and return contract response."""
    logger.info("Fit request received", extra={"series_id": series_id, "detector": detector})

    series = TimeSeries(
        data=[
            DataPoint(timestamp=timestamp, value=value)
            for timestamp, value in zip(payload.timestamps, payload.values)
        ]
    )
    trained = model_service.train(series_id=series_id, data=series, detector=detector)
    logger.info(
        "Fit request completed",
        extra={"series_id": series_id, "detector": detector, "version": trained.version},
    )
    return FitResponse(
        series_id=trained.series_id,
        detector=trained.detector,
        version=trained.version,
        points_used=trained.n_samples,
    )
