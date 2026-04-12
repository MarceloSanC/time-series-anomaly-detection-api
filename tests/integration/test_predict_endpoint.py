from __future__ import annotations

from fastapi.testclient import TestClient


def _fit_payload(start_timestamp: int, start_value: float) -> dict[str, list[float] | list[int]]:
    return {
        "timestamps": [start_timestamp + i for i in range(30)],
        "values": [start_value + (0.2 * i) for i in range(30)],
    }


def _train_baseline(client: TestClient) -> None:
    """Create two versions so prediction can resolve latest and explicit versions."""
    first = client.post(
        "/fit/sensor_A",
        json=_fit_payload(start_timestamp=1, start_value=10.0),
    )
    second = client.post(
        "/fit/sensor_A",
        json=_fit_payload(start_timestamp=101, start_value=20.0),
    )
    assert first.status_code == 200
    assert second.status_code == 200


def test_predict_endpoint_uses_latest_model_when_version_not_provided(client: TestClient) -> None:
    """Default prediction path should resolve to latest trained model version."""
    _train_baseline(client)

    response = client.post(
        "/predict/sensor_A",
        json={"timestamp": "1700000100", "value": 99.0},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_version"] == "v2"
    assert payload["anomaly"] is True
    assert isinstance(payload["anomaly"], bool)


def test_predict_endpoint_accepts_explicit_v2_query_param(client: TestClient) -> None:
    """Prediction should allow explicitly selecting version v2."""
    _train_baseline(client)

    response = client.post(
        "/predict/sensor_A?version=v2",
        json={"timestamp": "1700000201", "value": 15.0},
    )

    assert response.status_code == 200
    assert response.json()["model_version"] == "v2"


def test_predict_endpoint_keeps_old_versions_accessible_after_retrain(client: TestClient) -> None:
    """Retraining should keep previous versions addressable via query param."""
    _train_baseline(client)  # creates v1 and v2
    third = client.post(
        "/fit/sensor_A",
        json=_fit_payload(start_timestamp=201, start_value=30.0),  # creates v3
    )
    assert third.status_code == 200
    assert third.json()["version"] == "v3"

    old_version_response = client.post(
        "/predict/sensor_A?version=v1",
        json={"timestamp": "1700000202", "value": 15.0},
    )
    latest_version_response = client.post(
        "/predict/sensor_A",
        json={"timestamp": "1700000203", "value": 15.0},
    )

    assert old_version_response.status_code == 200
    assert latest_version_response.status_code == 200
    assert old_version_response.json()["model_version"] == "v1"
    assert latest_version_response.json()["model_version"] == "v3"


def test_predict_endpoint_returns_404_for_unknown_series(client: TestClient) -> None:
    """Unknown series_id should map to normalized 404 error payload."""
    response = client.post(
        "/predict/missing_series",
        json={"timestamp": "1700000300", "value": 5.0},
    )

    assert response.status_code == 404
    payload = response.json()
    assert payload["error"] == "SERIES_NOT_FOUND"
    assert "timestamp" in payload


def test_predict_endpoint_rejects_invalid_series_id(client: TestClient) -> None:
    """Unsafe series_id path segment must map to normalized 400 error payload."""
    response = client.post(
        "/predict/sensor..A",
        json={"timestamp": "1700000300", "value": 5.0},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"] == "INVALID_SERIES_ID"
    assert "timestamp" in payload
