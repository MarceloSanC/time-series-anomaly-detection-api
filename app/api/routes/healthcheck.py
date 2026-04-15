from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import get_metrics_service, get_model_service
from app.domain.schemas import ErrorResponse
from app.services.metrics_service import MetricsService
from app.services.model_service import ModelService

router = APIRouter(tags=["Health Check"])


class HealthMetrics(BaseModel):
    avg: float
    p95: float


class HealthcheckResponse(BaseModel):
    series_trained: int
    inference_latency_ms: HealthMetrics
    training_latency_ms: HealthMetrics


def _aggregate_metrics(snapshot: dict[str, Any], prefix: str) -> HealthMetrics:
    """Aggregate endpoint metrics into average and weighted p95 latency."""
    endpoints = snapshot.get("endpoints", {})
    if not isinstance(endpoints, dict):
        return HealthMetrics(avg=0.0, p95=0.0)

    total_requests = 0
    total_time_ms = 0.0
    weighted_p95_sum = 0.0
    for endpoint, metric in endpoints.items():
        if not isinstance(endpoint, str) or not endpoint.startswith(prefix):
            continue
        if not isinstance(metric, dict):
            continue

        count = int(metric.get("request_count", 0) or 0)
        total = float(metric.get("total_time_ms", 0.0) or 0.0)
        p95 = float(metric.get("p95_latency_ms", 0.0) or 0.0)

        total_requests += count
        total_time_ms += total
        weighted_p95_sum += p95 * count

    if total_requests == 0:
        return HealthMetrics(avg=0.0, p95=0.0)

    avg = total_time_ms / total_requests
    p95 = weighted_p95_sum / total_requests
    return HealthMetrics(avg=avg, p95=p95)


@router.get(
    "/healthcheck",
    response_model=HealthcheckResponse,
    summary="Service health and latency snapshot",
    description=(
        "Returns the count of trained series and aggregated latency metrics for training and prediction endpoints. "
        "Metrics are in-memory only — tracked from process startup, not persisted across restarts — "
        "and are computed from all requests received by each endpoint since the service started."
    ),
    responses={
        500: {
            "model": ErrorResponse,
            "description": "Unexpected internal error.",
            "content": {
                "application/json": {
                    "examples": {
                        "internal_error": {
                            "summary": "Unhandled exception",
                            "value": {
                                "error": "INTERNAL_ERROR",
                                "message": "An unexpected internal error occurred",
                                "detail": None,
                                "timestamp": "2026-04-15T18:00:00Z",
                            },
                        }
                    }
                }
            },
        }
    },
)
def healthcheck(
    model_service: ModelService = Depends(get_model_service),
    metrics_service: MetricsService = Depends(get_metrics_service),
) -> HealthcheckResponse:
    """Return service-level readiness and latency metrics snapshot."""
    series_trained = len(model_service.list_series())
    snapshot = metrics_service.snapshot()

    return HealthcheckResponse(
        series_trained=series_trained,
        inference_latency_ms=_aggregate_metrics(snapshot=snapshot, prefix="/predict"),
        training_latency_ms=_aggregate_metrics(snapshot=snapshot, prefix="/fit"),
    )
