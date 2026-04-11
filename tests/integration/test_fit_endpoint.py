from __future__ import annotations

from fastapi.testclient import TestClient


def test_fit_endpoint_trains_series_and_returns_contract_payload(client: TestClient) -> None:
    """Train a series via HTTP and validate response contract fields."""
    response = client.post(
        "/fit/sensor_A",
        json={
            "timestamps": [1700000001, 1700000002, 1700000003],
            "values": [10.0, 11.5, 12.0],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["series_id"] == "sensor_A"
    assert payload["version"] == "v1"
    assert payload["points_used"] == 3


def test_fit_endpoint_increments_version_on_retrain(client: TestClient) -> None:
    """Retraining the same series should increment model version labels."""
    first = client.post(
        "/fit/sensor_A",
        json={"timestamps": [1, 2, 3], "values": [1.0, 2.0, 3.0]},
    )
    second = client.post(
        "/fit/sensor_A",
        json={"timestamps": [4, 5, 6], "values": [2.0, 3.0, 4.0]},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["version"] == "v1"
    assert second.json()["version"] == "v2"
