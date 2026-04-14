from __future__ import annotations

import logging

import numpy as np

from app.config import settings
from app.domain.exceptions import (
    ConstantSeriesError,
    DuplicateTimestampsError,
    FlatLineDetectedError,
    InsufficientDataError,
    InvalidValuesError,
    TemporalGapDetectedError,
    UnorderedTimestampsError,
)
from app.domain.schemas import TimeSeries

logger = logging.getLogger(__name__)


class ValidationService:
    """Apply preflight business-rule validation before model training."""

    def __init__(
        self,
        min_data_points: int | None = None,
        std_threshold: float | None = None,
        flat_line_window: int | None = None,
        max_temporal_gap_factor: float | None = None,
    ) -> None:
        self.min_data_points = settings.min_data_points if min_data_points is None else min_data_points
        self.std_threshold = settings.std_threshold if std_threshold is None else std_threshold
        self.flat_line_window = settings.flat_line_window if flat_line_window is None else flat_line_window
        self.max_temporal_gap_factor = (
            settings.max_temporal_gap_factor if max_temporal_gap_factor is None else max_temporal_gap_factor
        )

    def validate_training_data(self, data: TimeSeries) -> None:
        """Validate series against all business rules in fail-fast order."""
        points = list(data.data)
        values = [point.value for point in points]
        timestamps = [point.timestamp for point in points]

        if len(points) < self.min_data_points:
            logger.warning(
                "Validation rejected: insufficient data",
                extra={"n_samples": len(points), "min_data_points": self.min_data_points},
            )
            raise InsufficientDataError(
                f"At least {self.min_data_points} data points are required; got {len(points)}"
            )

        if not bool(np.isfinite(values).all()):
            logger.warning("Validation rejected: invalid values")
            raise InvalidValuesError("Training data contains NaN or infinite values")

        std = float(np.std(values))
        if std < self.std_threshold:
            logger.warning(
                "Validation rejected: constant series",
                extra={"std": std, "std_threshold": self.std_threshold},
            )
            raise ConstantSeriesError(
                f"Series standard deviation {std} is below threshold {self.std_threshold}"
            )

        if len(set(timestamps)) != len(timestamps):
            logger.warning("Validation rejected: duplicate timestamps")
            raise DuplicateTimestampsError("Training data contains duplicate timestamps")

        if any(next_ts <= current_ts for current_ts, next_ts in zip(timestamps, timestamps[1:])):
            logger.warning("Validation rejected: unordered timestamps")
            raise UnorderedTimestampsError("Timestamps must be strictly increasing")

        if len(points) >= self.flat_line_window:
            trailing_values = values[-self.flat_line_window:]
            if max(trailing_values) == min(trailing_values):
                logger.warning(
                    "Validation rejected: flat-line trailing window",
                    extra={"flat_line_window": self.flat_line_window},
                )
                raise FlatLineDetectedError(
                    f"Trailing {self.flat_line_window} values are identical, possible frozen/disconnected sensor"
                )

        if len(points) < 2:
            return

        intervals = [next_ts - current_ts for current_ts, next_ts in zip(timestamps, timestamps[1:])]
        median_interval = float(np.median(intervals))
        max_gap = max(intervals)
        allowed_gap = self.max_temporal_gap_factor * median_interval
        if max_gap > allowed_gap:
            logger.warning(
                "Validation rejected: temporal gap detected",
                extra={
                    "max_gap": max_gap,
                    "median_interval": median_interval,
                    "max_temporal_gap_factor": self.max_temporal_gap_factor,
                },
            )
            raise TemporalGapDetectedError(
                f"Temporal gap {max_gap} exceeds allowed threshold {allowed_gap} "
                f"({self.max_temporal_gap_factor} x median interval {median_interval})"
            )
