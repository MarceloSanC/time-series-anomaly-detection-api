from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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
    description="Returns a detector-aware PNG image for training data visualization.",
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
    """Render detector-aware training plot as PNG."""
    logger.info("Plot request received", extra={"series_id": series_id, "version": version, "detector": detector})
    plot_data = model_service.get_plot_data(series_id=series_id, version=version, detector=detector)

    fig, ax = plt.subplots(figsize=(10, 4))
    if plot_data["detector"] == "isolation_forest":
        _render_isolation_forest_plot(fig=fig, ax=ax, plot_data=plot_data)
    else:
        _render_gaussian_plot(fig=fig, ax=ax, plot_data=plot_data)

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    logger.info(
        "Plot request completed",
        extra={"series_id": series_id, "version": plot_data["version"], "detector": detector},
    )
    return StreamingResponse(buffer, media_type="image/png")


def _render_gaussian_plot(fig: Figure, ax: Axes, plot_data: dict[str, Any]) -> None:
    training_data = plot_data["training_data"]
    dates = [datetime.fromtimestamp(int(p["timestamp"]), tz=UTC) for p in training_data]
    values = [float(p["value"]) for p in training_data]

    mean = float(plot_data["mean"])
    std = float(plot_data["std"])
    upper = mean + 3 * std
    lower = mean - 3 * std

    flags = plot_data.get("training_anomaly_flags")
    has_flags = isinstance(flags, list) and len(flags) == len(values)
    if has_flags:
        normal_d = [d for d, f in zip(dates, flags) if not f]
        normal_v = [v for v, f in zip(values, flags) if not f]
        anomaly_d = [d for d, f in zip(dates, flags) if f]
        anomaly_v = [v for v, f in zip(values, flags) if f]
        ax.scatter(normal_d, normal_v, s=14, alpha=0.8, color="#1f77b4", label="normal")
        if anomaly_d:
            ax.scatter(anomaly_d, anomaly_v, s=14, alpha=0.9, color="#d62728", label="anomaly")
    else:
        ax.scatter(dates, values, s=14, alpha=0.8, label="training points")

    ax.axhline(mean, linewidth=1.8, label="mean")
    ax.axhline(upper, linestyle="--", linewidth=1.4, label="+3 sigma")
    ax.axhline(lower, linestyle="--", linewidth=1.4, label="-3 sigma")
    _add_trend_line(ax=ax, training_data=training_data, dates=dates, values=values)
    ax.set_title(f"Series {plot_data['series_id']} ({plot_data['version']}) — {len(values)} points")
    _apply_axis_layout(fig=fig, ax=ax, dates=dates)


def _render_isolation_forest_plot(fig: Figure, ax: Axes, plot_data: dict[str, Any]) -> None:
    training_data = plot_data["training_data"]
    dates = [datetime.fromtimestamp(int(p["timestamp"]), tz=UTC) for p in training_data]
    values = [float(p["value"]) for p in training_data]

    training_scores = plot_data.get("training_scores")
    has_scores = (
        isinstance(training_scores, list)
        and len(training_scores) == len(values)
        and all(isinstance(score, (int, float)) for score in training_scores)
    )
    flags = plot_data.get("training_anomaly_flags")
    has_flags = isinstance(flags, list) and len(flags) == len(values)

    if has_scores:
        scatter = ax.scatter(
            dates,
            values,
            c=[float(score) for score in training_scores],
            cmap="coolwarm_r",
            s=16,
            alpha=0.85,
            label="training points",
        )
        fig.colorbar(scatter, ax=ax, label="anomaly score")
        if has_flags:
            flagged_d = [d for d, f in zip(dates, flags) if f]
            flagged_v = [v for v, f in zip(values, flags) if f]
            if flagged_d:
                ax.scatter(flagged_d, flagged_v, s=40, marker="x", color="black", linewidths=1.2, label="flagged", zorder=5)
    else:
        ax.scatter(dates, values, s=14, alpha=0.8, color="#1f77b4", label="training points")

    _add_trend_line(ax=ax, training_data=training_data, dates=dates, values=values)
    contamination = _format_contamination(plot_data.get("contamination"))
    score_threshold = plot_data.get("score_threshold")
    title = f"Series {plot_data['series_id']} ({plot_data['version']}) — {len(values)} points"
    if contamination is not None:
        title = f"{title} - contamination={contamination}"
    if isinstance(score_threshold, (int, float)):
        title = f"{title} - threshold={float(score_threshold):.3f}"
    ax.set_title(title)
    _apply_axis_layout(fig=fig, ax=ax, dates=dates)


def _add_trend_line(
    ax: Axes,
    training_data: list[dict[str, Any]],
    dates: list[datetime],
    values: list[float],
) -> None:
    timestamps = [int(p["timestamp"]) for p in training_data]
    coeffs = np.polyfit(timestamps, values, 1)
    trend = np.polyval(coeffs, timestamps).tolist()
    ax.plot(dates, trend, linestyle="--", linewidth=0.8, color="gray", alpha=0.6, label="trend")


def _format_contamination(raw_contamination: Any) -> str | None:
    if raw_contamination is None:
        return None
    if isinstance(raw_contamination, str):
        if raw_contamination == "auto":
            return "auto"
        return raw_contamination
    if isinstance(raw_contamination, (int, float)):
        return f"{float(raw_contamination):.1%}"
    return str(raw_contamination)


def _apply_axis_layout(fig: Figure, ax: Axes, dates: list[datetime]) -> None:
    is_single_day = all(dt.date() == dates[0].date() for dt in dates)
    ax.set_xlabel("timestamp")
    ax.set_ylabel("value")
    if is_single_day:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=UTC))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=UTC))
    ax.legend(loc="best")
    fig.tight_layout()
    fig.autofmt_xdate()
