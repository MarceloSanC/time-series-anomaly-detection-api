from __future__ import annotations

from fastapi.testclient import TestClient

from app.domain.models import AnomalyDetectionModel, IsolationForestDetector
from app.domain.schemas import DataPoint, TimeSeries


def _fit_payload(start_timestamp: int, start_value: float, n: int = 40) -> dict[str, list[float] | list[int]]:
    return {
        "timestamps": [start_timestamp + i for i in range(n)],
        "values": [start_value + (0.3 * i) for i in range(n)],
    }


def _varied_fit_payload(start_timestamp: int = 1, n: int = 100) -> dict[str, list[float] | list[int]]:
    """Payload with sufficient variance for IsolationForest training."""
    import math

    values = [10.0 + math.sin(i * 0.3) * 2 + (i % 7) * 0.1 for i in range(n)]
    return {
        "timestamps": [start_timestamp + i for i in range(n)],
        "values": values,
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


def test_plot_gaussian_with_explicit_detector_param_returns_png(client: TestClient) -> None:
    """`?detector=gaussian` should render correctly and exercise the blue/red anomaly coloring path."""
    trained = client.post("/fit/sensor_g", json=_fit_payload(start_timestamp=1, start_value=10.0))
    assert trained.status_code == 200

    response = client.get("/plot", params={"series_id": "sensor_g", "detector": "gaussian"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_plot_isolation_forest_returns_png(client: TestClient) -> None:
    """`?detector=isolation_forest` after IF training should render score-colored scatter as PNG."""
    trained = client.post(
        "/fit/sensor_if?detector=isolation_forest",
        json=_varied_fit_payload(start_timestamp=1),
    )
    assert trained.status_code == 200

    response = client.get("/plot", params={"series_id": "sensor_if", "detector": "isolation_forest"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_plot_isolation_forest_version_not_in_detector_namespace_returns_404(client: TestClient) -> None:
    """`?detector=isolation_forest&version=v1` where v1 only exists in gaussian namespace → 404."""
    trained = client.post("/fit/sensor_ns", json=_fit_payload(start_timestamp=1, start_value=10.0))
    assert trained.status_code == 200

    response = client.get(
        "/plot",
        params={"series_id": "sensor_ns", "detector": "isolation_forest", "version": "v1"},
    )

    assert response.status_code == 404
    payload = response.json()
    assert payload["error"] == "SERIES_NOT_FOUND"


def test_plot_isolation_forest_series_not_found_when_only_gaussian_trained(client: TestClient) -> None:
    """`?detector=isolation_forest` for a series trained only with gaussian → 404 SERIES_NOT_FOUND."""
    trained = client.post("/fit/sensor_go", json=_fit_payload(start_timestamp=1, start_value=10.0))
    assert trained.status_code == 200

    response = client.get("/plot", params={"series_id": "sensor_go", "detector": "isolation_forest"})

    assert response.status_code == 404
    payload = response.json()
    assert payload["error"] == "SERIES_NOT_FOUND"


def test_plot_unsupported_detector_returns_422(client: TestClient) -> None:
    """`?detector=random_forest` should return normalized 422 UNSUPPORTED_DETECTOR."""
    response = client.get("/plot", params={"series_id": "any", "detector": "random_forest"})

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"] == "UNSUPPORTED_DETECTOR"
    assert "random_forest" in payload["message"]
    assert "timestamp" in payload


def test_plot_gaussian_legacy_metadata_without_anomaly_flags_returns_png(client: TestClient) -> None:
    """Gaussian metadata without `training_anomaly_flags` should fall back to uniform scatter (no crash)."""
    model = AnomalyDetectionModel().fit(
        TimeSeries(data=[DataPoint(timestamp=i, value=float(i)) for i in range(1, 41)])
    )
    client.app.state.model_repository.save(
        series_id="sensor_legacy_g",
        version="v1",
        model=model,
        metadata={
            "version": "v1",
            "model_params": {"mean": 20.5, "std": 11.69},
            "n_samples": 40,
            "trained_at": "2026-01-01T00:00:00Z",
            "training_duration_ms": 1.0,
            "data_range": {"min_timestamp": 1, "max_timestamp": 40},
            "training_data": [{"timestamp": i, "value": float(i)} for i in range(1, 41)],
        },
    )

    response = client.get("/plot", params={"series_id": "sensor_legacy_g"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_plot_isolation_forest_legacy_metadata_without_scores_returns_png(client: TestClient) -> None:
    """IF metadata without `training_scores`/`training_anomaly_flags` should render with uniform color (no crash)."""
    import math

    values = [10.0 + math.sin(i * 0.3) for i in range(40)]
    model = IsolationForestDetector().fit(
        TimeSeries(data=[DataPoint(timestamp=i + 1, value=v) for i, v in enumerate(values)])
    )
    client.app.state.model_repository.save(
        series_id="sensor_legacy_if",
        version="v1",
        model=model,
        detector="isolation_forest",
        metadata={
            "version": "v1",
            "detector": "isolation_forest",
            "model_params": {
                "n_estimators": 100,
                "contamination": "auto",
                "score_threshold": model.score_threshold,
            },
            "n_samples": 40,
            "trained_at": "2026-01-01T00:00:00Z",
            "training_duration_ms": 1.0,
            "data_range": {"min_timestamp": 1, "max_timestamp": 40},
            "training_data": [{"timestamp": i + 1, "value": v} for i, v in enumerate(values)],
        },
    )

    response = client.get("/plot", params={"series_id": "sensor_legacy_if", "detector": "isolation_forest"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")
