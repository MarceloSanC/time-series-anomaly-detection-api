from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.exceptions import InvalidSeriesIdError
from app.domain.models import AnomalyDetectionModel
from app.domain.schemas import TimeSeries, DataPoint
from app.repository.model_repository import ModelRepository


def _fitted_model(values: list[float]) -> AnomalyDetectionModel:
    """Build and fit a model from a small list of values."""
    model = AnomalyDetectionModel()
    model.fit(TimeSeries(data=[DataPoint(timestamp=i + 1, value=v) for i, v in enumerate(values)]))
    return model


def _metadata(version: str, n_samples: int) -> dict[str, object]:
    """Create deterministic metadata payload used by repository tests."""
    return {
        "version": version,
        "mean": 10.0,
        "std": 2.0,
        "n_samples": n_samples,
        "trained_at": "2026-01-01T00:00:00Z",
        "training_duration_ms": 1.2,
        "data_range": {
            "min_timestamp": 1,
            "max_timestamp": n_samples,
        },
    }


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Ensure saved model artifacts can be loaded back without data loss."""
    repository = ModelRepository(storage_path=tmp_path)
    model = _fitted_model([9.0, 10.0, 11.0])

    repository.save(series_id="sensor_A", version="v1", model=model, metadata=_metadata("v1", 3))
    loaded_model, loaded_metadata = repository.load(series_id="sensor_A", version="v1")

    assert loaded_model.mean == pytest.approx(model.mean)
    assert loaded_model.std == pytest.approx(model.std)
    assert loaded_metadata["version"] == "v1"
    assert loaded_metadata["n_samples"] == 3


def test_save_writes_index_atomically_and_without_tmp_residue(tmp_path: Path) -> None:
    """Ensure index file is written atomically and temp residue is cleaned."""
    repository = ModelRepository(storage_path=tmp_path)

    repository.save(
        series_id="sensor_A",
        version="v1",
        model=_fitted_model([1.0, 2.0, 3.0]),
        metadata=_metadata("v1", 3),
    )

    series_dir = tmp_path / "sensor_A"
    index_path = series_dir / "index.json"
    tmp_index_path = series_dir / "index.json.tmp"

    assert index_path.exists()
    assert not tmp_index_path.exists()

    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["latest_version"] == "v1"
    assert index["versions"] == ["v1"]


def test_version_listing_and_exists(tmp_path: Path) -> None:
    """Validate version index order and version existence checks."""
    repository = ModelRepository(storage_path=tmp_path)

    repository.save(
        series_id="sensor_A",
        version="v1",
        model=_fitted_model([1.0, 2.0, 3.0]),
        metadata=_metadata("v1", 3),
    )
    repository.save(
        series_id="sensor_A",
        version="v2",
        model=_fitted_model([2.0, 3.0, 4.0]),
        metadata=_metadata("v2", 3),
    )

    index = repository.get_index("sensor_A")
    assert index is not None
    assert index["latest_version"] == "v2"
    assert index["versions"] == ["v1", "v2"]

    assert repository.version_exists("sensor_A", "v1") is True
    assert repository.version_exists("sensor_A", "v2") is True
    assert repository.version_exists("sensor_A", "v3") is False


def test_list_all_returns_all_series_indexes(tmp_path: Path) -> None:
    """Ensure repository listing returns index entries for every series."""
    repository = ModelRepository(storage_path=tmp_path)

    repository.save(
        series_id="sensor_A",
        version="v1",
        model=_fitted_model([1.0, 2.0, 3.0]),
        metadata=_metadata("v1", 3),
    )
    repository.save(
        series_id="sensor_B",
        version="v1",
        model=_fitted_model([4.0, 5.0, 6.0]),
        metadata=_metadata("v1", 3),
    )

    indexes = repository.list_all()

    assert len(indexes) == 2
    series_ids = sorted(index["series_id"] for index in indexes)
    assert series_ids == ["sensor_A", "sensor_B"]


@pytest.mark.parametrize(
    "series_id",
    [
        "../escape",
        r"sensor\\A",
        "sensor\x00A",
    ],
)
def test_invalid_series_id_is_rejected(tmp_path: Path, series_id: str) -> None:
    """Reject empty or unsafe series identifiers used in filesystem paths."""
    repository = ModelRepository(storage_path=tmp_path)

    with pytest.raises(InvalidSeriesIdError):
        repository.get_index(series_id)


def test_load_raises_file_not_found_for_missing_model(tmp_path: Path) -> None:
    """Raise when requested model artifacts are missing from storage."""
    repository = ModelRepository(storage_path=tmp_path)

    with pytest.raises(FileNotFoundError):
        repository.load(series_id="sensor_A", version="v1")
