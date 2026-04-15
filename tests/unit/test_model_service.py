from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.exceptions import (
    MetadataIncompleteError,
    PlotDataUnavailableError,
    SeriesNotFoundError,
    UnsupportedDetectorError,
    VersionNotFoundForDetectorError,
)
from app.domain.models import AnomalyDetectionModel
from app.domain.schemas import DataPoint, TimeSeries
from app.repository.model_repository import ModelRepository
from app.services.model_service import ModelService
from app.services.validation_service import ValidationService
from app.utils.concurrency import LockManager


def _series(values: list[float], start_ts: int = 1) -> TimeSeries:
    """Build a small ordered TimeSeries fixture from raw float values."""
    return TimeSeries(
        data=[DataPoint(timestamp=start_ts + idx, value=value) for idx, value in enumerate(values)]
    )


def test_train_returns_expected_payload_and_persists_model(tmp_path: Path) -> None:
    """Validate train response fields and successful persistence side effects."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(
        repository=repository,
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )

    response = service.train(series_id="sensor_A", data=_series([10.0, 12.0, 14.0, 16.0]))

    assert response.series_id == "sensor_A"
    assert response.version == "v1"
    assert response.n_samples == 4
    assert response.model_params["std"] > 0
    assert response.training_duration_ms >= 0
    assert response.trained_at.endswith("Z")


def test_predict_uses_latest_version_by_default(tmp_path: Path) -> None:
    """Ensure predict defaults to the latest available model version."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(
        repository=repository,
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )

    train_response = service.train(series_id="sensor_A", data=_series([10.0, 11.0, 12.0, 13.0]))

    prediction = service.predict(series_id="sensor_A", data_point=DataPoint(timestamp=100, value=99.0))

    assert prediction.series_id == "sensor_A"
    assert prediction.version == "v1"
    assert prediction.value == 99.0
    assert prediction.timestamp == 100
    assert prediction.model_params["mean"] == pytest.approx(train_response.model_params["mean"])
    assert prediction.model_params["upper_bound"] == pytest.approx(
        train_response.model_params["mean"] + 3 * train_response.model_params["std"]
    )
    assert isinstance(prediction.is_anomaly, bool)


def test_train_increments_version_on_retrain(tmp_path: Path) -> None:
    """Ensure retraining same series increments stored version label."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(
        repository=repository,
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )

    first = service.train(series_id="sensor_A", data=_series([1.0, 2.0, 3.0]))
    second = service.train(series_id="sensor_A", data=_series([2.0, 3.0, 4.0]))

    assert first.version == "v1"
    assert second.version == "v2"


def test_predict_raises_series_not_found(tmp_path: Path) -> None:
    """Raise SeriesNotFoundError when predicting an unknown series_id."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(
        repository=repository,
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )

    with pytest.raises(SeriesNotFoundError):
        service.predict(series_id="missing", data_point=DataPoint(timestamp=1, value=1.0))


def test_predict_raises_version_not_found_for_detector_on_missing_version(tmp_path: Path) -> None:
    """Raise VersionNotFoundForDetectorError when requested version does not exist."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(
        repository=repository,
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )

    service.train(series_id="sensor_A", data=_series([1.0, 2.0, 3.0]))

    with pytest.raises(VersionNotFoundForDetectorError):
        service.predict(series_id="sensor_A", data_point=DataPoint(timestamp=1, value=5.0), version="v999")


def test_predict_accepts_explicit_v2_version(tmp_path: Path) -> None:
    """Allow predictions against explicitly requested v2 model version."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(
        repository=repository,
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )

    service.train(series_id="sensor_A", data=_series([1.0, 2.0, 3.0]))
    service.train(series_id="sensor_A", data=_series([10.0, 11.0, 12.0]))

    prediction = service.predict(
        series_id="sensor_A",
        data_point=DataPoint(timestamp=99, value=20.0),
        version="v2",
    )

    assert prediction.version == "v2"


def test_get_series_info_returns_latest_metadata(tmp_path: Path) -> None:
    """Return latest version metadata for an existing series."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(
        repository=repository,
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )

    service.train(series_id="sensor_A", data=_series([1.0, 2.0, 3.0]))
    service.train(series_id="sensor_A", data=_series([2.0, 3.0, 4.0, 5.0, 6.0]))

    info = service.get_series_info("sensor_A")

    assert info.series_id == "sensor_A"
    assert info.latest_version == "v2"
    assert info.versions == ["v1", "v2"]
    assert info.n_samples == 5


def test_get_series_info_raises_series_not_found(tmp_path: Path) -> None:
    """Raise SeriesNotFoundError when querying info for unknown series."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(
        repository=repository,
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )

    with pytest.raises(SeriesNotFoundError):
        service.get_series_info("missing")


def test_get_plot_data_returns_training_points_for_latest_version(tmp_path: Path) -> None:
    """Return plot-ready metadata including training points for latest version."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(
        repository=repository,
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )

    service.train(series_id="sensor_A", data=_series([1.0, 2.0, 3.0]))
    service.train(series_id="sensor_A", data=_series([2.0, 3.0, 4.0]))

    plot_data = service.get_plot_data(series_id="sensor_A")

    assert plot_data["series_id"] == "sensor_A"
    assert plot_data["version"] == "v2"
    assert isinstance(plot_data["training_data"], list)
    assert len(plot_data["training_data"]) == 3


def test_get_plot_data_raises_when_training_points_missing(tmp_path: Path) -> None:
    """Raise PlotDataUnavailableError for legacy metadata without training_data."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(
        repository=repository,
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )

    model = AnomalyDetectionModel().fit(_series([1.0, 2.0, 3.0]))
    repository.save(
        series_id="sensor_A",
        version="v1",
        model=model,
        metadata={
            "version": "v1",
            "mean": 10.0,
            "std": 1.0,
            "n_samples": 3,
            "trained_at": "2026-01-01T00:00:00Z",
            "training_duration_ms": 0.1,
            "data_range": {"min_timestamp": 1, "max_timestamp": 3},
        },
    )

    with pytest.raises(PlotDataUnavailableError):
        service.get_plot_data(series_id="sensor_A")


def test_get_plot_data_raises_when_metadata_file_is_missing(tmp_path: Path) -> None:
    """Raise PlotDataUnavailableError when metadata.json no longer exists."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(
        repository=repository,
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )
    service.train(series_id="sensor_A", data=_series([1.0, 2.0, 3.0]))

    metadata_path = tmp_path / "sensor_A" / "gaussian" / "v1" / "metadata.json"
    metadata_path.unlink()

    with pytest.raises(PlotDataUnavailableError):
        service.get_plot_data(series_id="sensor_A")


def test_train_recovers_when_latest_version_label_is_malformed(tmp_path: Path) -> None:
    """Fallback to v1 when previous index latest_version cannot be parsed."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(
        repository=repository,
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )

    repository.save(
        series_id="sensor_A",
        version="vx",
        model=AnomalyDetectionModel().fit(_series([1.0, 2.0, 3.0])),
        metadata={
            "version": "vx",
            "mean": 2.0,
            "std": 1.0,
            "n_samples": 3,
            "trained_at": "2026-01-01T00:00:00Z",
            "training_duration_ms": 0.1,
            "data_range": {"min_timestamp": 1, "max_timestamp": 3},
            "training_data": [
                {"timestamp": 1, "value": 1.0},
                {"timestamp": 2, "value": 2.0},
                {"timestamp": 3, "value": 3.0},
            ],
        },
    )

    retrained = service.train(series_id="sensor_A", data=_series([4.0, 5.0, 6.0]))

    assert retrained.version == "v1"


def test_build_data_quality_with_normal_metadata_payload(tmp_path: Path) -> None:
    """Compute expected data-quality indicators from regular metadata input."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(
        repository=repository,
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )
    metadata = {
        "mean": 2.0,
        "std": 1.0,
        "data_range": {"min_timestamp": 10, "max_timestamp": 20},
        "training_data": [
            {"timestamp": 10, "value": 1.0},
            {"timestamp": 15, "value": 2.0},
            {"timestamp": 20, "value": 3.0},
        ],
    }

    quality = service._build_data_quality(metadata=metadata, n_samples=3)  # noqa: SLF001

    assert quality.n_samples == 3
    assert quality.mean == 2.0
    assert quality.std == 1.0
    assert quality.min_value == 1.0
    assert quality.max_value == 3.0
    assert quality.time_span_seconds == 10
    assert quality.points_per_second == 0.3


def test_build_data_quality_falls_back_when_training_data_missing(tmp_path: Path) -> None:
    """Fallback to mean bounds when metadata lacks training_data values."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(
        repository=repository,
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )
    metadata = {
        "mean": 5.0,
        "std": 2.0,
        "data_range": {"min_timestamp": 100, "max_timestamp": 100},
    }

    quality = service._build_data_quality(metadata=metadata, n_samples=7)  # noqa: SLF001

    assert quality.min_value == 5.0
    assert quality.max_value == 5.0
    assert quality.time_span_seconds == 0
    assert quality.points_per_second == 7.0


def test_list_model_summaries_strict_raises_for_missing_metadata(tmp_path: Path) -> None:
    """Strict mode should raise when latest metadata file is missing."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(
        repository=repository,
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )
    service.train(series_id="sensor_A", data=_series([1.0, 2.0, 3.0]))
    (tmp_path / "sensor_A" / "gaussian" / "v1" / "metadata.json").unlink()

    with pytest.raises(MetadataIncompleteError):
        service.list_model_summaries(strict=True)


def test_list_model_summaries_non_strict_skips_incomplete_series(tmp_path: Path) -> None:
    """Non-strict mode should skip incomplete series and return valid summaries."""
    repository = ModelRepository(storage_path=tmp_path)
    service = ModelService(
        repository=repository,
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )
    service.train(series_id="sensor_ok", data=_series([1.0, 2.0, 3.0]))
    service.train(series_id="sensor_bad", data=_series([4.0, 5.0, 6.0]))
    (tmp_path / "sensor_bad" / "gaussian" / "v1" / "metadata.json").unlink()

    summaries = service.list_model_summaries(strict=False)

    assert len(summaries) == 1
    assert summaries[0].series_id == "sensor_ok"


def _varied_series(n: int = 100, seed: int = 42) -> TimeSeries:
    """Return a normally-distributed series suitable for IsolationForest training."""
    import random
    rng = random.Random(seed)
    values = [10.0 + rng.gauss(0, 0.5) for _ in range(n)]
    return _series(values)


def test_train_default_detector_is_gaussian(tmp_path: Path) -> None:
    """train() with no detector argument must use gaussian and return correct model_params."""
    service = ModelService(
        repository=ModelRepository(storage_path=tmp_path),
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )
    response = service.train(series_id="sensor_A", data=_series([10.0, 11.0, 12.0, 13.0]))

    assert response.detector == "gaussian"
    assert response.model_params is not None
    assert "mean" in response.model_params
    assert "std" in response.model_params


def test_train_isolation_forest_dispatches_correctly(tmp_path: Path) -> None:
    """train(detector='isolation_forest') must store model and return score_threshold."""
    service = ModelService(
        repository=ModelRepository(storage_path=tmp_path),
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )
    response = service.train(series_id="sensor_A", data=_varied_series(), detector="isolation_forest")

    assert response.detector == "isolation_forest"
    assert response.model_params is not None
    assert "score_threshold" in response.model_params
    assert isinstance(response.model_params["score_threshold"], float)


def test_predict_isolation_forest_returns_correct_model_params(tmp_path: Path) -> None:
    """predict() for isolation_forest must return score_threshold in model_params."""
    service = ModelService(
        repository=ModelRepository(storage_path=tmp_path),
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )
    service.train(series_id="sensor_A", data=_varied_series(), detector="isolation_forest")
    prediction = service.predict(
        series_id="sensor_A",
        data_point=DataPoint(timestamp=1700001000, value=10.1),
        detector="isolation_forest",
    )

    assert prediction.detector == "isolation_forest"
    assert prediction.model_params is not None
    assert "score_threshold" in prediction.model_params
    assert isinstance(prediction.is_anomaly, bool)


def test_gaussian_and_isolation_forest_versions_are_independent(tmp_path: Path) -> None:
    """Training both detectors on the same series must not share version state."""
    service = ModelService(
        repository=ModelRepository(storage_path=tmp_path),
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )
    g1 = service.train(series_id="sensor_A", data=_series([10.0, 11.0, 12.0]), detector="gaussian")
    g2 = service.train(series_id="sensor_A", data=_series([13.0, 14.0, 15.0]), detector="gaussian")
    iso1 = service.train(series_id="sensor_A", data=_varied_series(), detector="isolation_forest")

    assert g1.version == "v1"
    assert g2.version == "v2"
    assert iso1.version == "v1"  # isolation_forest starts its own version sequence


def test_train_raises_unsupported_detector_error(tmp_path: Path) -> None:
    """train() must raise UnsupportedDetectorError before any repository call for unknown detectors."""
    service = ModelService(
        repository=ModelRepository(storage_path=tmp_path),
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )
    with pytest.raises(UnsupportedDetectorError):
        service.train(series_id="sensor_A", data=_series([1.0, 2.0, 3.0]), detector="random_forest")  # type: ignore[arg-type]


def test_predict_raises_unsupported_detector_error(tmp_path: Path) -> None:
    """predict() must raise UnsupportedDetectorError before any repository call for unknown detectors."""
    service = ModelService(
        repository=ModelRepository(storage_path=tmp_path),
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )
    with pytest.raises(UnsupportedDetectorError):
        service.predict(
            series_id="sensor_A",
            data_point=DataPoint(timestamp=1, value=1.0),
            detector="random_forest",  # type: ignore[arg-type]
        )


def test_predict_raises_version_not_found_for_detector(tmp_path: Path) -> None:
    """predict() must raise VersionNotFoundForDetectorError when requested version is missing."""
    service = ModelService(
        repository=ModelRepository(storage_path=tmp_path),
        lock_manager=LockManager(),
        validation_service=ValidationService(min_data_points=1),
    )
    service.train(series_id="sensor_A", data=_series([1.0, 2.0, 3.0]))

    with pytest.raises(VersionNotFoundForDetectorError):
        service.predict(series_id="sensor_A", data_point=DataPoint(timestamp=1, value=1.0), version="v999")
