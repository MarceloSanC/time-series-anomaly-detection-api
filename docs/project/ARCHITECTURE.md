# ARCHITECTURE.md — TECHNICAL DECISIONS AND TRADE-OFFS

This document records every significant architectural decision made for this project,
the alternatives considered, and the reasoning behind each choice.
GPT-codex and Claude must read this before proposing changes to any architectural pattern.
For model-focused failure analysis and detector evolution paths, see
`docs/project/MODELING_NOTES.md`.

---

## 1. PERSISTENCE: joblib on local filesystem

**Decision:** Store models as `joblib` files under `storage/{series_id}/{version}/model.joblib`.

**Alternatives considered:**
- SQLite with BLOB storage — adds a dependency, harder to inspect manually, no real benefit for this scale
- MLflow — massive operational overhead, overkill for a single-process service
- Redis / object storage — requires external services, violates zero-dependency-infra goal

**Trade-offs accepted:**
- Not suitable for horizontal scaling (multiple instances would not share state)
- Acceptable for single-instance deployment as specified

**Rules:**
- Always use `joblib.dump` / `joblib.load`, never `pickle` directly
- Store metadata separately in `metadata.json` so models can be inspected without deserialization
- Always write `index.json` atomically (write-then-rename)

---

## 2. CONCURRENCY: threading.Lock per series_id

**Decision:** Use `threading.Lock` per `series_id` inside a `LockManager`. Training is synchronous inside the lock.

**Alternatives considered:**
- `asyncio.Lock` — requires all code in the critical section to be async-compatible, which numpy/joblib are not
- `asyncio.run_in_executor` + thread pool — adds complexity (executor management, async/sync boundary) with no measurable benefit at this scale
- Global lock — kills throughput for concurrent training of different series

**Trade-offs accepted:**
- Training of the same `series_id` is serialized (no concurrent retraining of the same series)
- Training of different `series_id` values is fully parallel
- This is correct behavior: concurrent retraining of the same series would produce undefined version ordering

**Why not asyncio here:**
The FastAPI event loop is async, but training and I/O in this service are short operations (< 100ms for the statistical model provided). Pushing them to an executor gains nothing measurable and adds debugging complexity. Revisit only if training becomes CPU-intensive.

---

## 3. VERSIONING: incremental string versions

**Decision:** Versions are strings `v1`, `v2`, `v3`. An `index.json` file per series tracks the list and latest pointer.

**Alternatives considered:**
- Timestamp-based versions (e.g., `2024-01-15T10:30:00Z`) — hard to reference in URLs, no natural ordering guarantees
- UUID-based versions — not human-readable, harder to reference manually
- Semver — overkill for this use case

**Atomic write pattern:**
```python
import os
import json
import tempfile

def write_index_atomically(index_path: Path, data: dict) -> None:
    tmp_path = index_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, index_path)  # atomic on POSIX
```

**Why atomic:** A crash or power loss during a non-atomic write leaves a corrupted `index.json`, which breaks all subsequent operations on that series. The rename operation is atomic on POSIX systems.

---

## 4. METRICS: in-memory with bounded latency buffer

**Decision:** Track metrics in a Python dict with `threading.Lock`. Cap the latency sample buffer at 1000 entries (FIFO eviction). Export p50/p95/p99 on demand using `numpy.percentile`.

**Alternatives considered:**
- Prometheus + Grafana — correct long-term solution, but adds 2 external services to docker-compose
- StatsD — same problem
- Simple counter only — loses latency distribution, insufficient for evaluation criteria

**Trade-offs accepted:**
- Metrics are lost on restart (acceptable for this scope)
- p95/p99 computed over last 1000 requests, not all-time (prevents unbounded memory growth)

**Why bounded buffer:** An unbounded list of latency samples grows forever. At 1000 entries the p95/p99 is statistically stable and memory stays bounded.

---

## 5. VALIDATION: fail-fast with configurable thresholds

**Decision:** `ValidationService` runs before the lock is acquired in `model_service.train()`. Uses fail-fast: the first failing rule raises immediately without checking remaining rules.

**Why before the lock:** Validation is read-only and fast. Acquiring the lock, then validating, would block other callers of the same series unnecessarily during the validation period.

**Configurable thresholds:** `MIN_DATA_POINTS` and `STD_THRESHOLD` are in `config.py` and readable from `.env`. This prevents the validation from being too rigid in unexpected legitimate use cases.

---

## 6. API DESIGN: RESTful with consistent error contract

**Decision:** Follow `docs/context/openapi_spec.yaml` exactly for HTTP routes and payload contracts. Internal domain models may differ, but the API layer must map to/from the OpenAPI schema.

**Error response schema (non-negotiable):**
```json
{
  "error": "SERIES_NOT_FOUND",
  "message": "No model found for series_id 'sensor_XYZ'",
  "detail": null,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Status code mapping:** Follow the OpenAPI contract first; use internal error codes only when they map cleanly to the API responses.

---

## 7. ANOMALY DETECTION MODEL: use as provided

**Decision:** Use the `AnomalyDetectionModel` exactly as provided by the challenge specification. Fix the iteration bug silently (iterate over `data.data`, not `data`). Do not change the algorithm.

**Known limitation:** The model only detects positive anomalies (above `mean + 3*std`). Negative anomalies (below `mean - 3*std`) are not detected. This is a property of the provided algorithm. Document in README under Failure Modes. Do not patch.

**Why not patch:** The challenge explicitly provides this model as the model of choice. Changing the algorithm scope-creeps into model evaluation territory, which is explicitly out of scope per the spec.

---

## 8. LOGGING: correlation ID per request

**Decision:** Generate a `request_id` UUID in middleware, store in `contextvars.ContextVar`, inject into every log record via a `logging.Filter`.

**Why ContextVar and not thread-local:** FastAPI with `asyncio` can multiplex multiple requests on the same thread. Thread-local storage would produce incorrect `request_id` values. `ContextVar` is async-safe.

**Log format:** `%(asctime)s [%(request_id)s] %(levelname)s %(name)s: %(message)s`

---

## 9. PLOT ENDPOINT: matplotlib headless

**Decision:** `matplotlib.use('Agg')` must be called before any other matplotlib import. Load metadata from `metadata.json`, not the model, for efficiency.

**Critical Docker requirement:** In a Docker container without a display server, matplotlib will fail with a backend error unless `Agg` is set explicitly. Do not rely on `MPLBACKEND` env var — set it in code.

**Scope:** Scatter of training points (timestamp vs value), horizontal line at mean, dashed lines at mean ± 3*std. No external fonts, no elaborate styling.

---

## 10. DOCKER: single service, volume-mounted storage

**Decision:** Single `docker-compose.yml` with one service. Mount `./storage` as a volume for persistence.

**Non-root user:** Dockerfile must create and use a non-root user. Running as root in a container is a security antipattern.

**Why not multi-stage build for this size:** The service is small. A multi-stage build adds complexity without meaningful image size reduction. Use a single-stage build from `python:3.11-slim`.

---

## FAILURE MODES (document in README, do not fix unless noted)

| Failure Mode | Behavior | Fixable? |
|---|---|---|
| Positive-only anomaly detection | Values below mean-3σ not flagged | No — algorithm spec |
| In-memory metrics lost on restart | Metrics reset to zero | Acceptable for scope |
| Single-instance storage | Cannot scale horizontally | Out of scope |
| Low-variance but not constant series | May produce unreliable models | Mitigated by STD_THRESHOLD config |
| Large series training time | Blocks the series lock briefly | Acceptable — model is O(n) |
