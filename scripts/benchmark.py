from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
import numpy as np


def _build_training_payload(n_points: int, start_timestamp: int = 1_700_000_000) -> dict[str, list[float] | list[int]]:
    """Create a realistic training payload with mild trend and periodicity."""
    timestamps = [start_timestamp + i for i in range(n_points)]
    values = [50.0 + 0.02 * i + 2.5 * np.sin(i / 6.0) for i in range(n_points)]
    return {"timestamps": timestamps, "values": values}


async def _train_series(
    client: httpx.AsyncClient,
    base_url: str,
    series_id: str,
    n_points: int,
    detector: str,
) -> dict[str, Any]:
    payload = _build_training_payload(n_points=n_points)
    response = await client.post(
        f"{base_url}/fit/{series_id}",
        params={"detector": detector},
        json=payload,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


async def _predict_once(
    client: httpx.AsyncClient,
    base_url: str,
    series_id: str,
    timestamp: int,
    value: float,
    detector: str,
) -> float:
    """Perform one prediction request and return request latency in milliseconds."""
    started = perf_counter()
    response = await client.post(
        f"{base_url}/predict/{series_id}",
        params={"detector": detector},
        json={"timestamp": str(timestamp), "value": value},
        timeout=30.0,
    )
    response.raise_for_status()
    _ = response.json()
    return (perf_counter() - started) * 1000.0


def _latency_stats_ms(latencies_ms: list[float]) -> dict[str, float]:
    values = np.array(latencies_ms, dtype=float)
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "avg": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


async def run_benchmark(
    base_url: str,
    series_id: str,
    n_train_points: int,
    n_requests: int,
    detector: str,
    output_path: Path,
) -> dict[str, Any]:
    """Train baseline model and benchmark parallel inference latency/throughput."""
    async with httpx.AsyncClient() as client:
        train_response = await _train_series(
            client=client,
            base_url=base_url,
            series_id=series_id,
            n_points=n_train_points,
            detector=detector,
        )

        prediction_timestamp_base = 1_800_000_000
        prediction_value = 60.0
        bench_started = perf_counter()

        tasks = [
            _predict_once(
                client=client,
                base_url=base_url,
                series_id=series_id,
                timestamp=prediction_timestamp_base + idx,
                value=prediction_value,
                detector=detector,
            )
            for idx in range(n_requests)
        ]
        latencies_ms = await asyncio.gather(*tasks)
        total_seconds = perf_counter() - bench_started

    stats = _latency_stats_ms(latencies_ms)
    throughput_rps = float(n_requests / total_seconds) if total_seconds > 0 else 0.0

    result: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "target": {
            "base_url": base_url,
            "series_id": series_id,
            "detector": detector,
            "trained_version": train_response.get("version"),
        },
        "workload": {
            "n_train_points": n_train_points,
            "n_parallel_inference_requests": n_requests,
        },
        "latency_ms": stats,
        "throughput_rps": throughput_rps,
        "total_duration_seconds": float(total_seconds),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run parallel inference benchmark and save p50/p95/p99/throughput metrics.",
    )
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL.")
    parser.add_argument("--series-id", default="sensor_benchmark", help="Series id to train and benchmark.")
    parser.add_argument(
        "--detector",
        default="gaussian",
        choices=("gaussian", "isolation_forest"),
        help="Detector used in fit/predict benchmark path.",
    )
    parser.add_argument(
        "--both-detectors",
        action="store_true",
        help="Benchmark gaussian and isolation_forest sequentially and save combined output.",
    )
    parser.add_argument(
        "--n-train-points",
        type=int,
        default=180,
        help="Number of points used in fit before benchmark (must be > 100).",
    )
    parser.add_argument(
        "--n-requests",
        type=int,
        default=100,
        help="Number of parallel inference requests.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/benchmark_results.json"),
        help="Path to save benchmark json results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_train_points <= 100:
        raise SystemExit("n_train_points must be greater than 100")
    if args.n_requests <= 0:
        raise SystemExit("n_requests must be greater than 0")

    if args.both_detectors:
        results: dict[str, Any] = {"generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"), "runs": {}}
        for detector_name in ("gaussian", "isolation_forest"):
            run_result = asyncio.run(
                run_benchmark(
                    base_url=args.base_url.rstrip("/"),
                    series_id=f"{args.series_id}_{detector_name}",
                    n_train_points=args.n_train_points,
                    n_requests=args.n_requests,
                    detector=detector_name,
                    output_path=args.output,
                )
            )
            results["runs"][detector_name] = run_result
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
        print(json.dumps(results, indent=2, ensure_ascii=True))
        return

    result = asyncio.run(
        run_benchmark(
            base_url=args.base_url.rstrip("/"),
            series_id=args.series_id,
            n_train_points=args.n_train_points,
            n_requests=args.n_requests,
            detector=args.detector,
            output_path=args.output,
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
