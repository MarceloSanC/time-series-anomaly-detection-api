#!/usr/bin/env python3
"""
compare_detectors.py — Detector comparison script (Stage D).

Trains gaussian and isolation_forest via the HTTP API on the same
synthetic dataset, evaluates on a labeled held-out test split with
injected anomalies, and saves a comparison report to
scripts/detector_comparison.json.

Models are persisted after the run and can be inspected via:
  /plot?series_id=compare_gaussian                             (visualization)
  /models/compare_gaussian/versions/v1
  /models/compare_isolation_forest/versions/v1?include_data=true

Note: /plot only supports gaussian (mean/std bounds). For
isolation_forest use the ?include_data=true metadata endpoint.

Usage:
    .venv/bin/python scripts/compare_detectors.py
    .venv/bin/python scripts/compare_detectors.py --base-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
import numpy as np

# ── Dataset parameters ────────────────────────────────────────────────────────
_N_TRAIN = 150          # normal samples used to fit both detectors
_N_TEST_NORMAL = 40     # normal samples in the held-out test set
_N_TEST_ANOMALIES = 20  # anomalies injected into the held-out test set
_MEAN = 50.0
_STD = 5.0
# Anomalies injected at mean + _ANOMALY_SIGMA * std.
# Must exceed the gaussian 3-sigma threshold so both detectors get a fair
# opportunity. Only positive spikes so gaussian is not penalised for its
# known limitation of not flagging values below the mean.
_ANOMALY_SIGMA = 4.5
_RNG_SEED = 42
# ── Series IDs written to storage ─────────────────────────────────────────────
_SERIES: dict[str, str] = {
    "gaussian": "compare_gaussian",
    "isolation_forest": "compare_isolation_forest",
}
# ─────────────────────────────────────────────────────────────────────────────


def _build_dataset(
    rng: np.random.Generator,
) -> tuple[list[int], list[float], list[int], list[float], list[bool]]:
    """Build train and labeled held-out test splits.

    Returns:
        train_timestamps, train_values, test_timestamps, test_values, test_labels
    """
    train_values = rng.normal(loc=_MEAN, scale=_STD, size=_N_TRAIN)
    train_timestamps = list(range(1, _N_TRAIN + 1))

    normal_values = list(rng.normal(loc=_MEAN, scale=_STD, size=_N_TEST_NORMAL))
    # Positive spikes with tight spread so they reliably exceed the 3-sigma bound.
    anomaly_values = list(
        rng.normal(loc=_MEAN + _ANOMALY_SIGMA * _STD, scale=0.5, size=_N_TEST_ANOMALIES)
    )

    # Interleave anomalies at regular positions throughout the test set.
    test_values: list[float] = normal_values
    test_labels: list[bool] = [False] * _N_TEST_NORMAL
    step = max(_N_TEST_NORMAL // _N_TEST_ANOMALIES, 1)
    for idx, anom_val in enumerate(anomaly_values):
        pos = min(idx * step + step // 2, len(test_values))
        test_values.insert(pos, float(anom_val))
        test_labels.insert(pos, True)

    test_timestamps = list(range(_N_TRAIN + 1, _N_TRAIN + 1 + len(test_values)))

    return (
        train_timestamps,
        [float(v) for v in train_values],
        test_timestamps,
        [float(v) for v in test_values],
        test_labels,
    )


def _fit(
    client: httpx.Client,
    base_url: str,
    series_id: str,
    detector: str,
    timestamps: list[int],
    values: list[float],
) -> dict[str, Any]:
    """POST /fit/{series_id}?detector={detector}."""
    response = client.post(
        f"{base_url}/fit/{series_id}",
        params={"detector": detector},
        json={"timestamps": timestamps, "values": values},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def _evaluate(
    client: httpx.Client,
    base_url: str,
    series_id: str,
    detector: str,
    test_timestamps: list[int],
    test_values: list[float],
    test_labels: list[bool],
) -> dict[str, Any]:
    """POST /predict for each test point and compute detection metrics."""
    predictions: list[bool] = []
    latencies_ms: list[float] = []

    for ts, val in zip(test_timestamps, test_values):
        t0 = perf_counter()
        response = client.post(
            f"{base_url}/predict/{series_id}",
            params={"detector": detector},
            json={"timestamp": str(ts), "value": val},
            timeout=10.0,
        )
        latencies_ms.append((perf_counter() - t0) * 1000)
        response.raise_for_status()
        predictions.append(bool(response.json()["anomaly"]))

    n_pos = sum(test_labels)
    n_neg = len(test_labels) - n_pos
    tp = sum(p and l for p, l in zip(predictions, test_labels))
    fp = sum(p and not l for p, l in zip(predictions, test_labels))
    fn = sum(not p and l for p, l in zip(predictions, test_labels))
    tn = sum(not p and not l for p, l in zip(predictions, test_labels))

    tpr = tp / n_pos if n_pos else 0.0
    fpr = fp / n_neg if n_neg else 0.0

    lat = np.array(latencies_ms)
    return {
        "series_id": series_id,
        "n_train": _N_TRAIN,
        "n_test": len(test_values),
        "n_anomalies_in_test": n_pos,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "latency_ms": {
            "p50": round(float(np.percentile(lat, 50)), 4),
            "p95": round(float(np.percentile(lat, 95)), 4),
            "p99": round(float(np.percentile(lat, 99)), 4),
            "avg": round(float(np.mean(lat)), 4),
        },
    }


def _conclusion(detectors: dict[str, dict[str, Any]]) -> str:
    g = detectors["gaussian"]
    iso = detectors["isolation_forest"]
    tpr_winner = "gaussian" if g["tpr"] >= iso["tpr"] else "isolation_forest"
    fpr_winner = "gaussian" if g["fpr"] <= iso["fpr"] else "isolation_forest"
    lat_winner = (
        "gaussian"
        if g["latency_ms"]["p50"] <= iso["latency_ms"]["p50"]
        else "isolation_forest"
    )
    return (
        f"{tpr_winner} achieves higher TPR; "
        f"{fpr_winner} achieves lower FPR; "
        f"{lat_winner} is faster at p50 on this run. "
        "Gaussian detects only positive outliers (value > mean + 3*std) by design; "
        "anomalies are injected above mean + 4.5*std so both detectors compete fairly. "
        "IsolationForest requires sufficient training variance to avoid the masking effect "
        "(low-variance training data causes all points to score similarly). "
        "Latency includes HTTP round-trip overhead and is comparable between detectors "
        "but does not reflect pure algorithmic inference cost."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare gaussian vs isolation_forest detectors.")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "detector_comparison.json",
        help="Path for the JSON report output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_url = args.base_url.rstrip("/")

    rng = np.random.default_rng(_RNG_SEED)
    train_ts, train_vals, test_ts, test_vals, test_labels = _build_dataset(rng)

    n_anomalies = sum(test_labels)
    print(
        f"Dataset: {_N_TRAIN} train | "
        f"{len(test_vals)} test ({n_anomalies} anomalies at mean+{_ANOMALY_SIGMA}*std)",
        flush=True,
    )

    detector_results: dict[str, Any] = {}
    trained_versions: dict[str, str] = {}

    with httpx.Client() as client:
        for detector, series_id in _SERIES.items():
            print(f"\nFitting {detector} → series '{series_id}'...", flush=True)
            fit_resp = _fit(client, base_url, series_id, detector, train_ts, train_vals)
            trained_versions[detector] = str(fit_resp["version"])
            print(f"  Trained: version={fit_resp['version']} points={fit_resp['points_used']}", flush=True)

            print(f"Evaluating {detector} ({len(test_vals)} predictions)...", flush=True)
            metrics = _evaluate(client, base_url, series_id, detector, test_ts, test_vals, test_labels)
            detector_results[detector] = metrics
            print(
                f"  TPR={metrics['tpr']:.2%}  FPR={metrics['fpr']:.2%}"
                f"  p50={metrics['latency_ms']['p50']:.2f}ms",
                flush=True,
            )

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "dataset": {
            "description": (
                f"Synthetic industrial sensor data: gaussian-distributed normal samples "
                f"(mean={_MEAN}, std={_STD}, n_train={_N_TRAIN}) with positive spike "
                f"anomalies injected at mean+{_ANOMALY_SIGMA}*std in the test split."
            ),
            "n_train": _N_TRAIN,
            "n_test": len(test_vals),
            "n_anomalies_in_test": n_anomalies,
            "anomaly_injection": f"mean + {_ANOMALY_SIGMA} * std",
            "rng_seed": _RNG_SEED,
        },
        "inspect": {
            "gaussian_plot": f"{base_url}/plot?series_id=compare_gaussian",
            "gaussian_metadata": (
                f"{base_url}/models/compare_gaussian/versions/{trained_versions['gaussian']}"
            ),
            "isolation_forest_data": (
                f"{base_url}/models/compare_isolation_forest/versions/"
                f"{trained_versions['isolation_forest']}?include_data=true"
            ),
        },
        "detectors": detector_results,
        "conclusion": _conclusion(detector_results),
    }

    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    print(f"\nResults saved to {args.output}", file=sys.stderr)
    print(
        f"\nInspect trained models:\n"
        f"  /plot?series_id=compare_gaussian\n"
        f"  /models/compare_isolation_forest/versions/v1?include_data=true",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
