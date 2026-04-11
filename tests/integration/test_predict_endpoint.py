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


def test_predict_endpoint_accepts_explicit_version_query_param(client: TestClient) -> None:
    """Prediction should honor explicit version query parameter."""
    _train_baseline(client)

    response = client.post(
        "/predict/sensor_A?version=v1",
        json={"timestamp": "1700000200", "value": 15.0},
    )

    assert response.status_code == 200
    assert response.json()["model_version"] == "v1"


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
