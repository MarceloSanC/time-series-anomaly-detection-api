from __future__ import annotations

from fastapi.testclient import TestClient

from app.domain.models import AnomalyDetectionModel
from app.domain.schemas import DataPoint, TimeSeries


def _fit_payload(start_timestamp: int, start_value: float) -> dict[str, list[float] | list[int]]:
    return {
        "timestamps": [start_timestamp + i for i in range(40)],
        "values": [start_value + (0.3 * i) for i in range(40)],
    }


def test_models_endpoint_returns_empty_list_when_no_series_exist(client: TestClient) -> None:
    """`/models` should return an empty list when storage has no tracked series."""
    response = client.get("/models")

    assert response.status_code == 200
    assert response.json() == []


def test_models_endpoint_lists_multiple_series_summaries(client: TestClient) -> None:
    """`/models` should return one summary item per tracked series."""
    first = client.post("/fit/sensor_A", json=_fit_payload(start_timestamp=1, start_value=10.0))
    second = client.post("/fit/sensor_B", json=_fit_payload(start_timestamp=101, start_value=20.0))
    assert first.status_code == 200
    assert second.status_code == 200

    response = client.get("/models")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    series_ids = sorted(item["series_id"] for item in payload)
    assert series_ids == ["sensor_A", "sensor_B"]
    assert all(set(item.keys()) == {"series_id", "latest_version", "n_samples", "trained_at"} for item in payload)


def test_models_endpoint_tolerates_incomplete_metadata_by_default(client: TestClient) -> None:
    """Default `/models` should skip series with missing latest metadata and return valid ones."""
    valid = client.post("/fit/sensor_ok", json=_fit_payload(start_timestamp=1, start_value=10.0))
    invalid = client.post("/fit/sensor_bad", json=_fit_payload(start_timestamp=101, start_value=20.0))
    assert valid.status_code == 200
    assert invalid.status_code == 200

    missing_metadata = client.app.state.model_repository.storage_path / "sensor_bad" / "v1" / "metadata.json"
    missing_metadata.unlink()

    response = client.get("/models")

    assert response.status_code == 200
    payload = response.json()
    assert [item["series_id"] for item in payload] == ["sensor_ok"]


def test_models_endpoint_strict_mode_returns_422_for_incomplete_metadata(client: TestClient) -> None:
    """`/models?strict=true` should fail-fast when any latest metadata is missing."""
    valid = client.post("/fit/sensor_ok", json=_fit_payload(start_timestamp=1, start_value=10.0))
    invalid = client.post("/fit/sensor_bad", json=_fit_payload(start_timestamp=101, start_value=20.0))
    assert valid.status_code == 200
    assert invalid.status_code == 200

    missing_metadata = client.app.state.model_repository.storage_path / "sensor_bad" / "v1" / "metadata.json"
    missing_metadata.unlink()

    response = client.get("/models", params={"strict": "true"})

    assert response.status_code == 422
    payload = response.json()
    assert isinstance(payload, dict)
    assert payload["error"] == "INCOMPLETE_MODEL_METADATA"


def test_model_detail_endpoint_returns_expected_payload_with_data_quality(client: TestClient) -> None:
    """`/models/{series_id}` should return lineage data plus derived data_quality block."""
    trained = client.post("/fit/sensor_detail", json=_fit_payload(start_timestamp=1, start_value=10.0))
    assert trained.status_code == 200

    response = client.get("/models/sensor_detail")

    assert response.status_code == 200
    payload = response.json()
    assert payload["series_id"] == "sensor_detail"
    assert payload["latest_version"] == "v1"
    assert payload["versions"] == ["v1"]
    assert payload["n_samples"] == 40
    quality = payload["data_quality"]
    assert set(quality.keys()) == {
        "n_samples",
        "mean",
        "std",
        "min_value",
        "max_value",
        "time_span_seconds",
        "points_per_second",
    }
    assert quality["time_span_seconds"] >= 0
    assert quality["points_per_second"] == quality["n_samples"] / max(quality["time_span_seconds"], 1)
    assert quality["min_value"] <= quality["mean"] <= quality["max_value"]


def test_model_detail_endpoint_returns_404_for_unknown_series(client: TestClient) -> None:
    """Unknown series should map to normalized SERIES_NOT_FOUND payload."""
    response = client.get("/models/missing_series")

    assert response.status_code == 404
    payload = response.json()
    assert payload["error"] == "SERIES_NOT_FOUND"


def test_model_version_endpoint_returns_summary_without_training_data_by_default(client: TestClient) -> None:
    """`/models/{series}/versions/{version}` should exclude training_data by default."""
    trained = client.post("/fit/sensor_versions", json=_fit_payload(start_timestamp=1, start_value=10.0))
    assert trained.status_code == 200

    response = client.get("/models/sensor_versions/versions/v1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == "v1"
    assert "training_data" not in payload


def test_model_version_endpoint_includes_training_data_when_requested(client: TestClient) -> None:
    """`include_data=true` should include persisted training_data in response."""
    trained = client.post("/fit/sensor_versions", json=_fit_payload(start_timestamp=1, start_value=10.0))
    assert trained.status_code == 200

    response = client.get("/models/sensor_versions/versions/v1", params={"include_data": "true"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == "v1"
    assert isinstance(payload["training_data"], list)
    assert len(payload["training_data"]) == 40


def test_model_version_endpoint_include_data_returns_empty_list_for_legacy_metadata(client: TestClient) -> None:
    """Legacy metadata without training_data should return an empty list when include_data=true."""
    model = AnomalyDetectionModel().fit(
        TimeSeries(
            data=[
                DataPoint(timestamp=1, value=1.0),
                DataPoint(timestamp=2, value=2.0),
                DataPoint(timestamp=3, value=3.0),
            ]
        )
    )
    client.app.state.model_repository.save(
        series_id="sensor_legacy",
        version="v1",
        model=model,
        metadata={
            "version": "v1",
            "mean": 2.0,
            "std": 1.0,
            "n_samples": 3,
            "trained_at": "2026-01-01T00:00:00Z",
            "training_duration_ms": 1.0,
            "data_range": {"min_timestamp": 1, "max_timestamp": 3},
        },
    )

    response = client.get("/models/sensor_legacy/versions/v1", params={"include_data": "true"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["training_data"] == []


def test_model_version_endpoint_returns_404_for_unknown_version(client: TestClient) -> None:
    """Unknown model version should map to normalized VERSION_NOT_FOUND payload."""
    trained = client.post("/fit/sensor_versions", json=_fit_payload(start_timestamp=1, start_value=10.0))
    assert trained.status_code == 200

    response = client.get("/models/sensor_versions/versions/v999")

    assert response.status_code == 404
    payload = response.json()
    assert payload["error"] == "VERSION_NOT_FOUND"


def test_model_detail_data_quality_handles_zero_time_span(client: TestClient) -> None:
    """Data-quality calculation should use divisor max(time_span_seconds, 1)."""
    model = AnomalyDetectionModel().fit(
        TimeSeries(
            data=[
                DataPoint(timestamp=1, value=1.0),
                DataPoint(timestamp=2, value=2.0),
                DataPoint(timestamp=3, value=3.0),
            ]
        )
    )
    client.app.state.model_repository.save(
        series_id="sensor_zero_span",
        version="v1",
        model=model,
        metadata={
            "version": "v1",
            "mean": 2.0,
            "std": 1.0,
            "n_samples": 3,
            "trained_at": "2026-01-01T00:00:00Z",
            "training_duration_ms": 1.0,
            "data_range": {"min_timestamp": 10, "max_timestamp": 10},
            "training_data": [
                {"timestamp": 10, "value": 1.0},
                {"timestamp": 10, "value": 2.0},
                {"timestamp": 10, "value": 3.0},
            ],
        },
    )

    response = client.get("/models/sensor_zero_span")

    assert response.status_code == 200
    quality = response.json()["data_quality"]
    assert quality["time_span_seconds"] == 0
    assert quality["points_per_second"] == 3.0
