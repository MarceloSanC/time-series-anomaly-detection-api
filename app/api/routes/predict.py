from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, field_validator

from app.dependencies import get_model_service
from app.domain.schemas import DataPoint
from app.services.model_service import ModelService

router = APIRouter(tags=["Prediction"])
logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    timestamp: str
    value: float

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_string(cls, value: str) -> str:
        """Accept only non-empty values coercible to unix-timestamp integer."""
        text = value.strip()
        try:
            int(text)
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp must be a unix timestamp string")
        if text == "":
            raise ValueError("timestamp must be a unix timestamp string")
        return text


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    anomaly: bool
    model_version: str


@router.post("/predict/{series_id}", response_model=PredictResponse)
def predict_series(
    series_id: str,
    payload: PredictRequest,
    version: str | None = Query(default=None),
    model_service: ModelService = Depends(get_model_service),
) -> PredictResponse:
    """Predict anomaly status for one point using latest or requested version."""
    logger.info("Predict request received", extra={"series_id": series_id, "version": version})
    timestamp = int(payload.timestamp)

    prediction = model_service.predict(
        series_id=series_id,
        data_point=DataPoint(timestamp=timestamp, value=payload.value),
        version=version,
    )
    logger.info(
        "Predict request completed",
        extra={"series_id": series_id, "version": prediction.version, "anomaly": prediction.is_anomaly},
    )
    return PredictResponse(anomaly=prediction.is_anomaly, model_version=prediction.version)
