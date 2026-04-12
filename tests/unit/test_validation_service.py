from __future__ import annotations

import pytest

from app.domain.exceptions import (
    ConstantSeriesError,
    DuplicateTimestampsError,
    InsufficientDataError,
    InvalidValuesError,
    UnorderedTimestampsError,
)
from app.domain.schemas import DataPoint, TimeSeries
from app.services.validation_service import ValidationService


def _series(timestamps: list[int], values: list[float]) -> TimeSeries:
    """Build a TimeSeries fixture from explicit timestamp/value pairs."""
    assert len(timestamps) == len(values), (
        "timestamps and values must have the same length: "
        f"{len(timestamps)} != {len(values)}"
    )
    return TimeSeries(
        data=[DataPoint(timestamp=timestamp, value=value) for timestamp, value in zip(timestamps, values)]
    )


def test_validate_training_data_accepts_minimum_points_boundary() -> None:
    """Accept data when sample count is exactly the configured minimum."""
    validator = ValidationService(min_data_points=3, std_threshold=0.01)
    data = _series(timestamps=[1, 2, 3], values=[1.0, 2.0, 3.0])

    validator.validate_training_data(data)


def test_validate_training_data_rejects_insufficient_points() -> None:
    """Reject data when sample count is below configured minimum."""
    validator = ValidationService(min_data_points=4, std_threshold=0.01)
    data = _series(timestamps=[1, 2, 3], values=[1.0, 2.0, 3.0])

    with pytest.raises(InsufficientDataError):
        validator.validate_training_data(data)


def test_validate_training_data_accepts_non_constant_series() -> None:
    """Accept series when standard deviation is above threshold."""
    validator = ValidationService(min_data_points=3, std_threshold=0.2)
    data = _series(timestamps=[1, 2, 3], values=[1.0, 2.0, 3.0])

    validator.validate_training_data(data)


def test_validate_training_data_rejects_constant_series() -> None:
    """Reject series when standard deviation is below configured threshold."""
    validator = ValidationService(min_data_points=3, std_threshold=0.01)
    data = _series(timestamps=[1, 2, 3], values=[5.0, 5.0, 5.0])

    with pytest.raises(ConstantSeriesError):
        validator.validate_training_data(data)


def test_validate_training_data_accepts_unique_timestamps() -> None:
    """Accept training data when timestamps are all unique."""
    validator = ValidationService(min_data_points=4, std_threshold=0.01)
    data = _series(timestamps=[10, 20, 30, 40], values=[1.0, 2.0, 3.0, 4.0])

    validator.validate_training_data(data)


def test_validate_training_data_rejects_duplicate_timestamps() -> None:
    """Reject training data containing duplicated timestamps."""
    validator = ValidationService(min_data_points=4, std_threshold=0.01)
    data = _series(timestamps=[10, 20, 20, 30], values=[1.0, 2.0, 3.0, 4.0])

    with pytest.raises(DuplicateTimestampsError):
        validator.validate_training_data(data)


def test_validate_training_data_accepts_strictly_increasing_timestamps() -> None:
    """Accept training data when timestamps are strictly increasing."""
    validator = ValidationService(min_data_points=4, std_threshold=0.01)
    data = _series(timestamps=[1, 2, 3, 4], values=[2.0, 3.0, 4.0, 5.0])

    validator.validate_training_data(data)


def test_validate_training_data_rejects_unordered_timestamps() -> None:
    """Reject training data when timestamps are not strictly increasing."""
    validator = ValidationService(min_data_points=4, std_threshold=0.01)
    data = _series(timestamps=[1, 3, 2, 4], values=[2.0, 3.0, 4.0, 5.0])

    with pytest.raises(UnorderedTimestampsError):
        validator.validate_training_data(data)


def test_validate_training_data_accepts_finite_values() -> None:
    """Accept training data containing only finite numeric values."""
    validator = ValidationService(min_data_points=3, std_threshold=0.01)
    data = _series(timestamps=[1, 2, 3], values=[10.0, 11.0, 12.0])

    validator.validate_training_data(data)


def test_validate_training_data_rejects_non_finite_values() -> None:
    """Reject training data containing NaN or infinity."""
    validator = ValidationService(min_data_points=3, std_threshold=0.01)
    data = _series(timestamps=[1, 2, 3], values=[10.0, float("nan"), 12.0])

    with pytest.raises(InvalidValuesError):
        validator.validate_training_data(data)


def test_validate_training_data_rejects_infinite_values() -> None:
    """Reject training data containing infinite numeric values."""
    validator = ValidationService(min_data_points=3, std_threshold=0.01)
    data = _series(timestamps=[1, 2, 3], values=[10.0, float("inf"), 12.0])

    with pytest.raises(InvalidValuesError):
        validator.validate_training_data(data)
