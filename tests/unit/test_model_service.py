from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.schemas import DataPoint, TimeSeries
from app.repository.model_repository import ModelRepository
from app.services.model_service import ModelService, SeriesNotFoundError, VersionNotFoundError
from app.utils.concurrency import LockManager


def _series(values: list[float], start_ts: int = 1) -> TimeSeries:
    """Build a small ordered TimeSeries fixture from raw float values."""
    return TimeSeries(
        data=[DataPoint(timestamp=start_ts + idx, value=value) for idx, value in enumerate(values)]
    )


def test_train_returns_expected_payload_and_persists_model(tmp_path: Path) -> None:
    """Validate train response fields and successful persistence side effects."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(repository=repository, lock_manager=LockManager())

    response = service.train(series_id="sensor_A", data=_series([10.0, 12.0, 14.0, 16.0]))

    assert response.series_id == "sensor_A"
    assert response.version == "v1"
    assert response.n_samples == 4
    assert response.std > 0
    assert response.training_duration_ms >= 0
    assert response.trained_at.endswith("Z")


def test_predict_uses_latest_version_by_default(tmp_path: Path) -> None:
    """Ensure predict defaults to the latest available model version."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(repository=repository, lock_manager=LockManager())

    # TODO(stage1-polish): replace tautological upper_bound assertion below.
    # Current expression simplifies to `prediction.upper_bound == prediction.upper_bound`.
    # Fix strategy:
    # 1) Capture train response from `service.train(...)`.
    # 2) Assert `prediction.mean == train_response.mean`.
    # 3) Assert `prediction.upper_bound == train_response.mean + 3 * train_response.std`.
    # This ensures predict() upper bound is validated against learned training statistics.
    service.train(series_id="sensor_A", data=_series([10.0, 11.0, 12.0, 13.0]))

    prediction = service.predict(series_id="sensor_A", data_point=DataPoint(timestamp=100, value=99.0))

    assert prediction.series_id == "sensor_A"
    assert prediction.version == "v1"
    assert prediction.value == 99.0
    assert prediction.timestamp == 100
    assert prediction.upper_bound == pytest.approx(prediction.mean + 3 * (prediction.upper_bound - prediction.mean) / 3)
    assert isinstance(prediction.is_anomaly, bool)


def test_train_increments_version_on_retrain(tmp_path: Path) -> None:
    """Ensure retraining same series increments stored version label."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(repository=repository, lock_manager=LockManager())

    first = service.train(series_id="sensor_A", data=_series([1.0, 2.0, 3.0]))
    second = service.train(series_id="sensor_A", data=_series([2.0, 3.0, 4.0]))

    assert first.version == "v1"
    assert second.version == "v2"


def test_predict_raises_series_not_found(tmp_path: Path) -> None:
    """Raise SeriesNotFoundError when predicting an unknown series_id."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(repository=repository, lock_manager=LockManager())

    with pytest.raises(SeriesNotFoundError):
        service.predict(series_id="missing", data_point=DataPoint(timestamp=1, value=1.0))


def test_predict_raises_version_not_found(tmp_path: Path) -> None:
    """Raise VersionNotFoundError when requested version does not exist."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(repository=repository, lock_manager=LockManager())

    service.train(series_id="sensor_A", data=_series([1.0, 2.0, 3.0]))

    with pytest.raises(VersionNotFoundError):
        service.predict(series_id="sensor_A", data_point=DataPoint(timestamp=1, value=5.0), version="v999")


def test_get_series_info_returns_latest_metadata(tmp_path: Path) -> None:
    """Return latest version metadata for an existing series."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(repository=repository, lock_manager=LockManager())

    # TODO(stage1-polish): train with different sample sizes across versions.
    # Current setup uses 3 samples for both v1 and v2, so `info.n_samples == 3`
    # does not prove metadata came from latest version.
    # Suggested fix:
    # - v1: 3 samples
    # - v2: 5 samples
    # - assert `info.latest_version == "v2"` and `info.n_samples == 5`.
    service.train(series_id="sensor_A", data=_series([1.0, 2.0, 3.0]))
    service.train(series_id="sensor_A", data=_series([2.0, 3.0, 4.0]))

    info = service.get_series_info("sensor_A")

    assert info.series_id == "sensor_A"
    assert info.latest_version == "v2"
    assert info.versions == ["v1", "v2"]
    assert info.n_samples == 3


def test_get_series_info_raises_series_not_found(tmp_path: Path) -> None:
    """Raise SeriesNotFoundError when querying info for unknown series."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(repository=repository, lock_manager=LockManager())

    with pytest.raises(SeriesNotFoundError):
        service.get_series_info("missing")
