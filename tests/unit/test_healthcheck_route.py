from __future__ import annotations

from app.api.routes.healthcheck import _aggregate_metrics


def test_aggregate_metrics_returns_zeroes_when_endpoints_is_not_dict() -> None:
    """Guard against malformed snapshot payloads."""
    aggregated = _aggregate_metrics(snapshot={"endpoints": []}, prefix="/predict")

    assert aggregated.avg == 0.0
    assert aggregated.p95 == 0.0


def test_aggregate_metrics_ignores_non_dict_metric_entries() -> None:
    """Skip malformed endpoint metric objects while aggregating valid ones."""
    snapshot = {
        "endpoints": {
            "/predict/a": "not-a-dict",
            "/predict/b": {"request_count": 2, "total_time_ms": 10.0, "p95_latency_ms": 6.0},
        }
    }

    aggregated = _aggregate_metrics(snapshot=snapshot, prefix="/predict")

    assert aggregated.avg == 5.0
    assert aggregated.p95 == 6.0
