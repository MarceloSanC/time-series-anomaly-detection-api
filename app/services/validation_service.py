from __future__ import annotations

import logging

import numpy as np

from app.config import settings
from app.domain.exceptions import (
    ConstantSeriesError,
    DuplicateTimestampsError,
    InsufficientDataError,
    InvalidValuesError,
    UnorderedTimestampsError,
)
from app.domain.schemas import TimeSeries

logger = logging.getLogger(__name__)


class ValidationService:
    """Apply preflight business-rule validation before model training."""

    def __init__(self, min_data_points: int | None = None, std_threshold: float | None = None) -> None:
        self.min_data_points = settings.min_data_points if min_data_points is None else min_data_points
        self.std_threshold = settings.std_threshold if std_threshold is None else std_threshold

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

        if not bool(np.isfinite(values).all()):
            logger.warning("Validation rejected: invalid values")
            raise InvalidValuesError("Training data contains NaN or infinite values")
