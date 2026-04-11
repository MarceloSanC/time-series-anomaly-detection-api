from __future__ import annotations

from fastapi.testclient import TestClient


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
