from __future__ import annotations

from fastapi.testclient import TestClient

from app.dependencies import get_model_service
from app.domain.exceptions import ValidationServiceError


def _fit_payload(start_timestamp: int, start_value: float) -> dict[str, list[float] | list[int]]:
    return {
        "timestamps": [start_timestamp + i for i in range(30)],
        "values": [start_value + (0.2 * i) for i in range(30)],
    }


def test_fit_endpoint_trains_series_and_returns_contract_payload(client: TestClient) -> None:
    """Train a series via HTTP and validate response contract fields."""
    response = client.post(
        "/fit/sensor_A",
        json=_fit_payload(start_timestamp=1700000001, start_value=10.0),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["series_id"] == "sensor_A"
    assert payload["detector"] == "gaussian"
    assert payload["version"] == "v1"
    assert payload["points_used"] == 30


def test_fit_endpoint_increments_version_on_retrain(client: TestClient) -> None:
    """Retraining the same series should increment model version labels."""
    first = client.post(
        "/fit/sensor_A",
        json=_fit_payload(start_timestamp=1, start_value=1.0),
    )
    second = client.post(
        "/fit/sensor_A",
        json=_fit_payload(start_timestamp=101, start_value=2.0),
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["version"] == "v1"
    assert second.json()["version"] == "v2"


def test_fit_endpoint_rejects_invalid_series_id(client: TestClient) -> None:
    """Unsafe series_id path segment must map to normalized 400 error payload."""
    response = client.post(
        "/fit/sensor..A",
        json=_fit_payload(start_timestamp=1, start_value=1.0),
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"] == "INVALID_SERIES_ID"
    assert "timestamp" in payload


def test_fit_endpoint_rejects_empty_arrays_with_validation_error(client: TestClient) -> None:
    """Empty timestamps/values should be normalized as request validation error."""
    response = client.post("/fit/sensor_A", json={"timestamps": [], "values": []})

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"] == "VALIDATION_ERROR"


def test_fit_endpoint_rejects_mismatched_arrays_with_validation_error(client: TestClient) -> None:
    """Mismatched timestamps/values lengths should return normalized 422 payload."""
    response = client.post("/fit/sensor_A", json={"timestamps": [1, 2], "values": [10.0]})

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"] == "VALIDATION_ERROR"


def test_fit_endpoint_maps_unmapped_validation_service_error(client: TestClient) -> None:
    """Unmapped ValidationServiceError should fallback to generic VALIDATION_ERROR."""

    class BrokenService:
        def train(self, series_id: str, data: object, detector: str = "gaussian") -> object:
            _ = (series_id, data, detector)
            raise ValidationServiceError("custom validation failure")

    client.app.dependency_overrides[get_model_service] = lambda: BrokenService()
    try:
        response = client.post(
            "/fit/sensor_A",
            json=_fit_payload(start_timestamp=1, start_value=1.0),
        )
    finally:
        client.app.dependency_overrides.pop(get_model_service, None)

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"] == "VALIDATION_ERROR"


def test_fit_endpoint_accepts_isolation_forest_detector(client: TestClient) -> None:
    """Training with ?detector=isolation_forest should return detector-aware payload."""
    response = client.post(
        "/fit/sensor_iso?detector=isolation_forest",
        json=_fit_payload(start_timestamp=1700000001, start_value=10.0),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["series_id"] == "sensor_iso"
    assert payload["detector"] == "isolation_forest"
    assert payload["version"] == "v1"


def test_fit_endpoint_returns_422_for_unsupported_detector(client: TestClient) -> None:
    """Unsupported detector must map to UNSUPPORTED_DETECTOR via service validation."""
    response = client.post(
        "/fit/sensor_A?detector=random_forest",
        json=_fit_payload(start_timestamp=1700000001, start_value=10.0),
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"] == "UNSUPPORTED_DETECTOR"
    assert "is not supported" in payload["message"]
    assert "timestamp" in payload


def test_fit_endpoint_rejects_case_variant_detector_value(client: TestClient) -> None:
    """Detector values are case-sensitive and must be normalized by caller."""
    response = client.post(
        "/fit/sensor_A?detector=Gaussian",
        json=_fit_payload(start_timestamp=1700000001, start_value=10.0),
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"] == "UNSUPPORTED_DETECTOR"
    assert "Gaussian" in payload["message"]
