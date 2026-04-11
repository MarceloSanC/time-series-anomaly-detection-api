from __future__ import annotations

from fastapi.testclient import TestClient


def _fit_payload(start_timestamp: int, start_value: float) -> dict[str, list[float] | list[int]]:
    return {
        "timestamps": [start_timestamp + i for i in range(30)],
        "values": [start_value + (0.2 * i) for i in range(30)],
    }


def test_healthcheck_endpoint_returns_contract_payload(client: TestClient) -> None:
    """Healthcheck should expose required contract fields and metric groups."""
    response = client.get("/healthcheck")

    assert response.status_code == 200
    payload = response.json()
    assert payload["series_trained"] == 0
    assert set(payload["inference_latency_ms"].keys()) == {"avg", "p95"}
    assert set(payload["training_latency_ms"].keys()) == {"avg", "p95"}


def test_healthcheck_reflects_trained_series_and_non_negative_metrics(client: TestClient) -> None:
    """Healthcheck should reflect activity after fit/predict requests."""
    client.post(
        "/fit/sensor_A",
        json=_fit_payload(start_timestamp=1, start_value=10.0),
    )
    client.post(
        "/predict/sensor_A",
        json={"timestamp": "1700000400", "value": 20.0},
    )

    response = client.get("/healthcheck")

    assert response.status_code == 200
    payload = response.json()
    assert payload["series_trained"] == 1
    assert payload["inference_latency_ms"]["avg"] >= 0.0
    assert payload["inference_latency_ms"]["p95"] >= 0.0
    assert payload["training_latency_ms"]["avg"] >= 0.0
    assert payload["training_latency_ms"]["p95"] >= 0.0
