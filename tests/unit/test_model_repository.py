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

    detector_dir = tmp_path / "sensor_A" / "gaussian"
    index_path = detector_dir / "index.json"
    tmp_index_path = detector_dir / "index.json.tmp"

    assert index_path.exists()
    assert not tmp_index_path.exists()

    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["schema_version"] == "1"
    assert index["detector"] == "gaussian"
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


def test_load_metadata_returns_metadata_without_loading_model(tmp_path: Path) -> None:
    """Load only metadata for an existing series/version pair."""
    repository = ModelRepository(storage_path=tmp_path)
    repository.save(
        series_id="sensor_A",
        version="v1",
        model=_fitted_model([1.0, 2.0, 3.0]),
        metadata=_metadata("v1", 3),
    )

    metadata = repository.load_metadata(series_id="sensor_A", version="v1")

    assert metadata["version"] == "v1"
    assert metadata["n_samples"] == 3


def test_load_metadata_raises_file_not_found_when_missing(tmp_path: Path) -> None:
    """Raise when metadata file does not exist for requested series/version."""
    repository = ModelRepository(storage_path=tmp_path)

    with pytest.raises(FileNotFoundError):
        repository.load_metadata(series_id="sensor_A", version="v1")


def test_list_all_skips_non_directory_and_series_without_index(tmp_path: Path) -> None:
    """Ignore non-directory entries and folders that do not contain index.json."""
    repository = ModelRepository(storage_path=tmp_path)

    (tmp_path / "README.txt").write_text("ignore me", encoding="utf-8")
    (tmp_path / "orphan_series").mkdir(parents=True, exist_ok=True)
    repository.save(
        series_id="sensor_A",
        version="v1",
        model=_fitted_model([1.0, 2.0, 3.0]),
        metadata=_metadata("v1", 3),
    )

    indexes = repository.list_all()

    assert len(indexes) == 1
    assert indexes[0]["series_id"] == "sensor_A"


def test_detector_scoped_paths_are_independent(tmp_path: Path) -> None:
    """Two detectors on the same series must not share indexes or version state."""
    repository = ModelRepository(storage_path=tmp_path)
    model = _fitted_model([1.0, 2.0, 3.0])

    repository.save(series_id="sensor_A", version="v1", model=model, metadata=_metadata("v1", 3), detector="gaussian")
    repository.save(
        series_id="sensor_A", version="v1", model=model, metadata=_metadata("v1", 3), detector="isolation_forest"
    )
    repository.save(series_id="sensor_A", version="v2", model=model, metadata=_metadata("v2", 3), detector="gaussian")

    gaussian_index = repository.get_index("sensor_A", detector="gaussian")
    iso_index = repository.get_index("sensor_A", detector="isolation_forest")

    assert gaussian_index is not None
    assert gaussian_index["latest_version"] == "v2"
    assert gaussian_index["versions"] == ["v1", "v2"]
    assert gaussian_index["detector"] == "gaussian"

    assert iso_index is not None
    assert iso_index["latest_version"] == "v1"
    assert iso_index["versions"] == ["v1"]
    assert iso_index["detector"] == "isolation_forest"


def test_version_exists_is_scoped_per_detector(tmp_path: Path) -> None:
    """version_exists must check within the correct detector namespace."""
    repository = ModelRepository(storage_path=tmp_path)
    model = _fitted_model([1.0, 2.0, 3.0])

    repository.save(series_id="sensor_A", version="v1", model=model, metadata=_metadata("v1", 3), detector="gaussian")

    assert repository.version_exists("sensor_A", "v1", detector="gaussian") is True
    assert repository.version_exists("sensor_A", "v1", detector="isolation_forest") is False


def test_list_all_returns_one_entry_per_series_detector_pair(tmp_path: Path) -> None:
    """list_all must return one index per (series_id, detector) combination."""
    repository = ModelRepository(storage_path=tmp_path)
    model = _fitted_model([1.0, 2.0, 3.0])

    repository.save(series_id="sensor_A", version="v1", model=model, metadata=_metadata("v1", 3), detector="gaussian")
    repository.save(
        series_id="sensor_A", version="v1", model=model, metadata=_metadata("v1", 3), detector="isolation_forest"
    )
    repository.save(series_id="sensor_B", version="v1", model=model, metadata=_metadata("v1", 3), detector="gaussian")

    indexes = repository.list_all()

    assert len(indexes) == 3
    keys = sorted((idx["series_id"], idx["detector"]) for idx in indexes)
    assert keys == [("sensor_A", "gaussian"), ("sensor_A", "isolation_forest"), ("sensor_B", "gaussian")]


def test_index_contains_schema_version(tmp_path: Path) -> None:
    """Saved index must include schema_version field for forward compatibility."""
    repository = ModelRepository(storage_path=tmp_path)
    repository.save(
        series_id="sensor_A", version="v1", model=_fitted_model([1.0, 2.0, 3.0]), metadata=_metadata("v1", 3)
    )

    index = repository.get_index("sensor_A")

    assert index is not None
    assert index["schema_version"] == "1"


def test_load_is_scoped_per_detector(tmp_path: Path) -> None:
    """load() must read from the correct detector namespace and not cross into another."""
    repository = ModelRepository(storage_path=tmp_path)
    model = _fitted_model([1.0, 2.0, 3.0])

    repository.save(series_id="sensor_A", version="v1", model=model, metadata=_metadata("v1", 3), detector="gaussian")

    # Artifact exists under gaussian but not isolation_forest.
    loaded_model, loaded_meta = repository.load("sensor_A", "v1", detector="gaussian")
    assert loaded_meta["version"] == "v1"

    with pytest.raises(FileNotFoundError):
        repository.load("sensor_A", "v1", detector="isolation_forest")


def test_load_metadata_is_scoped_per_detector(tmp_path: Path) -> None:
    """load_metadata() must not read across detector namespaces."""
    repository = ModelRepository(storage_path=tmp_path)

    repository.save(
        series_id="sensor_A", version="v1", model=_fitted_model([1.0, 2.0, 3.0]), metadata=_metadata("v1", 3),
        detector="gaussian",
    )

    meta = repository.load_metadata("sensor_A", "v1", detector="gaussian")
    assert meta["n_samples"] == 3

    with pytest.raises(FileNotFoundError):
        repository.load_metadata("sensor_A", "v1", detector="isolation_forest")


def test_unsafe_detector_string_is_rejected(tmp_path: Path) -> None:
    """Detector strings with path traversal tokens must raise ValueError."""
    repository = ModelRepository(storage_path=tmp_path)
    model = _fitted_model([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        repository.save(series_id="sensor_A", version="v1", model=model, metadata=_metadata("v1", 3), detector="../x")

    with pytest.raises(ValueError):
        repository.get_index("sensor_A", detector="../x")
