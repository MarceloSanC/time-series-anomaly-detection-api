from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock
from time import monotonic
from typing import Any

import numpy as np


class MetricsService:
    def __init__(self, latency_window_size: int = 1000) -> None:
        self._request_count: dict[str, int] = defaultdict(int)
        self._total_time_ms: dict[str, float] = defaultdict(float)
        self._latencies_ms: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=latency_window_size)
        )
        self._error_count: dict[str, int] = defaultdict(int)
        self._start_time = monotonic()
        self._lock = Lock()

    def record(self, endpoint: str, duration_ms: float, error: bool = False) -> None:
        with self._lock:
            self._request_count[endpoint] += 1
            self._total_time_ms[endpoint] += duration_ms

            latencies = self._latencies_ms[endpoint]
            latencies.append(duration_ms)
            if error:
                self._error_count[endpoint] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            output: dict[str, dict[str, float | int | None]] = {}
            for endpoint, count in self._request_count.items():
                total = self._total_time_ms[endpoint]
                latencies = self._latencies_ms[endpoint]
                mean = total / count if count else None

                output[endpoint] = {
                    "request_count": count,
                    "error_count": self._error_count[endpoint],
                    "total_time_ms": total,
                    "mean_latency_ms": mean,
                    "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
                    "p99_latency_ms": float(np.percentile(latencies, 99)) if latencies else None,
                }
            return {
                "uptime_seconds": monotonic() - self._start_time,
                "endpoints": output,
            }
