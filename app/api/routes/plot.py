from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
import logging

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.dependencies import get_model_service
from app.domain.schemas import ErrorResponse
from app.services.model_service import ModelService

router = APIRouter(tags=["Visualization"])
logger = logging.getLogger(__name__)


@router.get(
    "/plot",
    summary="Render model training plot",
    description="Returns a PNG image with training points, mean line and ±3 sigma bounds.",
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Series or requested version not found.",
            "content": {
                "application/json": {
                    "examples": {
                        "series_not_found": {
                            "summary": "Series does not exist",
                            "value": {
                                "error": "SERIES_NOT_FOUND",
                                "message": "Series 'sensor_XYZ' not found",
                                "detail": None,
                                "timestamp": "2026-04-15T18:00:00Z",
                            },
                        },
                        "version_not_found_for_detector": {
                            "summary": "Version missing in selected detector namespace",
                            "value": {
                                "error": "VERSION_NOT_FOUND_FOR_DETECTOR",
                                "message": "Version 'v999' not found for series 'sensor_XYZ' detector 'gaussian'",
                                "detail": None,
                                "timestamp": "2026-04-15T18:00:00Z",
                            },
                        },
                    }
                }
            },
        },
        422: {
            "model": ErrorResponse,
            "description": "Plot data unavailable for selected model version or unsupported detector value.",
            "content": {
                "application/json": {
                    "examples": {
                        "plot_data_unavailable": {
                            "summary": "Training data not persisted for this version",
                            "value": {
                                "error": "PLOT_DATA_UNAVAILABLE",
                                "message": "Plot data not available for series 'sensor_XYZ' version 'v1'",
                                "detail": None,
                                "timestamp": "2026-04-15T18:00:00Z",
                            },
                        },
                        "unsupported_detector": {
                            "summary": "Unsupported detector query value",
                            "value": {
                                "error": "UNSUPPORTED_DETECTOR",
                                "message": "Detector 'random_forest' is not supported",
                                "detail": None,
                                "timestamp": "2026-04-15T18:00:00Z",
                            },
                        },
                    }
                }
            },
        },
    },
)
def plot_series(
    series_id: str = Query(..., description="Series identifier used during training."),
    version: str | None = Query(default=None, description="Optional model version (defaults to latest)."),
    detector: str = Query(
        default="gaussian",
        description="Detector type to use for plot data lookup. Supported: gaussian, isolation_forest.",
        openapi_examples={
            "gaussian": {"summary": "Default detector", "value": "gaussian"},
            "isolation_forest": {"summary": "Isolation Forest detector", "value": "isolation_forest"},
        },
    ),
    model_service: ModelService = Depends(get_model_service),
) -> StreamingResponse:
    """Render training points plus mean and 3-sigma bounds as PNG."""
    logger.info("Plot request received", extra={"series_id": series_id, "version": version, "detector": detector})
    plot_data = model_service.get_plot_data(series_id=series_id, version=version, detector=detector)

    training_data = plot_data["training_data"]
    timestamps = [int(point["timestamp"]) for point in training_data]
    dates = [datetime.fromtimestamp(timestamp, tz=UTC) for timestamp in timestamps]
    values = [float(point["value"]) for point in training_data]
    mean = float(plot_data["mean"])
    std = float(plot_data["std"])
    upper = mean + 3 * std
    lower = mean - 3 * std
    is_single_day = all(dt.date() == dates[0].date() for dt in dates)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(dates, values, s=14, alpha=0.8, label="training points")
    ax.axhline(mean, linewidth=1.8, label="mean")
    ax.axhline(upper, linestyle="--", linewidth=1.4, label="+3 sigma")
    ax.axhline(lower, linestyle="--", linewidth=1.4, label="-3 sigma")
    ax.set_title(f"Series {series_id} ({plot_data['version']})")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("value")
    if is_single_day:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=UTC))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=UTC))
    ax.legend(loc="best")
    fig.tight_layout()
    fig.autofmt_xdate()

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    logger.info(
        "Plot request completed",
        extra={"series_id": series_id, "version": plot_data["version"], "detector": detector},
    )
    return StreamingResponse(buffer, media_type="image/png")
