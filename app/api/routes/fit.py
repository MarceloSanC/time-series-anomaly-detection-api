from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_model_service
from app.domain.schemas import DataPoint, TimeSeries
from app.services.model_service import ModelService

router = APIRouter(tags=["Training"])
logger = logging.getLogger(__name__)


class FitRequest(BaseModel):
    timestamps: list[int] = Field(..., description="Unix timestamps for each observed value.")
    values: list[float] = Field(..., description="Observed values aligned with timestamps.")


class FitResponse(BaseModel):
    series_id: str
    version: str
    points_used: int


@router.post("/fit/{series_id}", response_model=FitResponse)
def fit_series(
    series_id: str,
    payload: FitRequest,
    model_service: ModelService = Depends(get_model_service),
) -> FitResponse:
    logger.info("Fit request received", extra={"series_id": series_id})

    if not payload.timestamps:
        raise HTTPException(status_code=422, detail="timestamps and values cannot be empty")

    if len(payload.timestamps) != len(payload.values):
        raise HTTPException(status_code=422, detail="timestamps and values must have the same length")

    series = TimeSeries(
        data=[
            DataPoint(timestamp=timestamp, value=value)
            for timestamp, value in zip(payload.timestamps, payload.values)
        ]
    )
    trained = model_service.train(series_id=series_id, data=series)
    logger.info("Fit request completed", extra={"series_id": series_id, "version": trained.version})
    return FitResponse(series_id=trained.series_id, version=trained.version, points_used=trained.n_samples)
