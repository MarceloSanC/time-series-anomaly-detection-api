from __future__ import annotations

from fastapi.testclient import TestClient

from app.domain.models import AnomalyDetectionModel
from app.domain.schemas import DataPoint, TimeSeries


def _fit_payload(start_timestamp: int, start_value: float) -> dict[str, list[float] | list[int]]:
    return {
        "timestamps": [start_timestamp + i for i in range(40)],
        "values": [start_value + (0.3 * i) for i in range(40)],
    }


def test_plot_endpoint_returns_png_for_latest_version(client: TestClient) -> None:
    """Plot endpoint should render PNG using latest trained model metadata."""
    trained = client.post("/fit/sensor_plot", json=_fit_payload(start_timestamp=1, start_value=10.0))
    assert trained.status_code == 200

    response = client.get("/plot", params={"series_id": "sensor_plot"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_plot_endpoint_accepts_explicit_version(client: TestClient) -> None:
    """Plot endpoint should allow selecting a concrete version in query param."""
    first = client.post("/fit/sensor_plot", json=_fit_payload(start_timestamp=1, start_value=10.0))
    second = client.post("/fit/sensor_plot", json=_fit_payload(start_timestamp=101, start_value=20.0))
    assert first.status_code == 200
    assert second.status_code == 200

    response = client.get("/plot", params={"series_id": "sensor_plot", "version": "v1"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_plot_endpoint_returns_422_when_training_data_is_missing(client: TestClient) -> None:
    """Legacy metadata without training_data should return normalized 422 payload."""
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

    response = client.get("/plot", params={"series_id": "sensor_legacy"})

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"] == "PLOT_DATA_UNAVAILABLE"
    assert "timestamp" in payload


def test_plot_endpoint_supports_multi_day_timestamp_window(client: TestClient) -> None:
    """Plot rendering should also work when training timestamps span multiple days."""
    timestamps = [1_700_000_000 + (i * 86_400) for i in range(40)]
    values = [10.0 + (0.5 * i) for i in range(40)]
    trained = client.post("/fit/sensor_plot_days", json={"timestamps": timestamps, "values": values})
    assert trained.status_code == 200

    response = client.get("/plot", params={"series_id": "sensor_plot_days"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")
