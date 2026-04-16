# ARCHITECTURE.md — TECHNICAL DECISIONS AND TRADE-OFFS

This document records every significant architectural decision made for this project,
the alternatives considered, and the reasoning behind each choice.
GPT-codex and Claude must read this before proposing changes to any architectural pattern.
For model-focused failure analysis and detector evolution paths, see
`docs/project/MODEL_DESIGN_NOTES.md`.

---

## 1. PERSISTENCE: joblib on local filesystem

**Decision:** Store models as `joblib` files under detector-scoped paths:
`storage/{series_id}/{detector}/{version}/model.joblib`.

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
- Keep one `index.json` per `(series_id, detector)` namespace
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

**Decision:** Versions are strings `v1`, `v2`, `v3`, scoped per detector namespace.
An `index.json` file per `(series_id, detector)` tracks the list and latest pointer.

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

## 8. PLOT ENDPOINT: matplotlib headless

**Decision:** `matplotlib.use('Agg')` must be called before any other matplotlib import. Load metadata from `metadata.json`, not the model, for efficiency.

**Critical Docker requirement:** In a Docker container without a display server, matplotlib will fail with a backend error unless `Agg` is set explicitly. Do not rely on `MPLBACKEND` env var — set it in code.

**Scope:** Scatter of training points (timestamp vs value), horizontal line at mean, dashed lines at mean ± 3*std. No external fonts, no elaborate styling.

---

## ARCHITECTURE EXTENSIONS (Post-Challenge Scope)

> Sections in this block cover extensions and planned evolution beyond the original challenge scope.

---

## 9. LOGGING: correlation ID per request

> **Status: Implemented**

**Decision:** Generate a `request_id` UUID in middleware, store in `contextvars.ContextVar`, inject into every log record via a `logging.Filter`.

**Why ContextVar and not thread-local:** FastAPI with `asyncio` can multiplex multiple requests on the same thread. Thread-local storage would produce incorrect `request_id` values. `ContextVar` is async-safe.

**Log format:** `%(asctime)s [%(request_id)s] %(levelname)s %(name)s: %(message)s`

---

## 10. DOCKER: single service, volume-mounted storage

> **Status: Implemented**

**Decision:** Single `docker-compose.yml` with one service. Mount `./storage` as a volume for persistence.

**Non-root user:** Dockerfile must create and use a non-root user. Running as root in a container is a security antipattern.

**Why not multi-stage build for this size:** The service is small. A multi-stage build adds complexity without meaningful image size reduction. Use a single-stage build from `python:3.11-slim`.

---

## 11. VALIDATION SERVICE: extension points and design boundaries

> **Status: Implemented**

**Decision:** `ValidationService` uses an imperative fail-fast chain of seven rules (five core + two optional sensor-quality rules). New rules require exactly two changes: a typed exception subclassing `ValidationServiceError` in `app/domain/exceptions.py`, and one entry in `VALIDATION_ERROR_CODE_MAP` in `app/api/error_handlers.py`. No handler functions, route handlers, or service signatures change.

**Why opt-in flags instead of always-on:** The sensor-quality rules (flat-line, temporal-gap) target industrial IoT failure modes outside the original challenge scope. Defaulting them to active would reject valid evaluator data. Boolean flags in `config.py` keep the feature demonstrable without side effects; all thresholds and flags are injectable via the constructor for test isolation.

**Modularization path:** If the sensor data quality rules grow beyond the current scope, `ValidationService` would be split into a dedicated `SensorDataQualityValidator` module. See Section 12 for the production context and domain motivation behind this boundary.

---

## 12. SENSOR DATA QUALITY: production context for industrial deployments

> **Status: Implemented**

**Decision:** The flat-line and temporal-gap validation rules are derived from real failure modes in industrial IoT sensor networks. Flat-line patterns indicate frozen sensors or disconnected signal paths; irregular temporal gaps indicate sampling instability or upstream pipeline failures. Both produce structurally valid data (numeric, finite, ordered) that yields unreliable models.

**Why opt-in and not always-on:** These rules are outside the original challenge scope and target a specific deployment context. Always-on defaults would silently reject training data that is valid in non-industrial settings. See Section 11 for the flag mechanism.

**Modularization trigger:** If sensor-quality rules exceed 3–4 rules, `ValidationService` should be split: a dedicated `SensorDataQualityValidator` handles domain-specific quality checks while `ValidationService` retains only structural integrity rules. The `TimeSeries` interface is the stable contract between them — no architectural rewrites required.

---

## 13. PRODUCTION EVOLUTION: streaming inference and industrial deployment

**Streaming topology (planned):** `ModelService.predict()` is already stateless and accepts a single `DataPoint` — no changes to the service layer are required to migrate from synchronous HTTP to event-driven inference:

```
Sensor → Kafka (raw_sensor_data) → Consumer → ModelService.predict() → Kafka (anomaly_events)
```

The consumer replaces the HTTP route handler as the entry point. The service layer is transport-agnostic by design.

**`series_id` → equipment mapping:** Each `series_id` maps to one sensor channel on one physical asset. A motor with three measurement axes would have three series: `motor_001_vibration_x`, `motor_001_vibration_y`, `motor_001_temperature`. The versioning and persistence model requires no changes for this mapping.

**Latency optimization path:** At 1 Hz sensor sampling, the inference SLA is < 1000ms. The current p99 (~300ms on a local single-instance setup) is within budget for single-sensor alerting but tightens under multi-sensor fan-out. 
Optimization: pre-load the latest model version into `app.state` on startup, eliminating `joblib.load()` from the critical path. Expected p99 after in-memory caching: < 5ms.

---

## 14. MULTI-DETECTOR API CONTRACT (Additive Extension)

> **Status: Implemented**

**Decision:** `/fit`, `/predict`, and `/models*` accept optional `?detector=`.
When omitted, the system resolves within `gaussian` namespace by default.

**Supported detector values:** `gaussian`, `isolation_forest`

**Error normalization:** invalid detector values return `422 UNSUPPORTED_DETECTOR`.
Version lookups that miss within selected detector namespace return
`404 VERSION_NOT_FOUND_FOR_DETECTOR`.

**Compatibility note:** this is additive behavior layered on top of the original challenge API.
Core request/response contracts remain backward-compatible for callers that omit `?detector=`.

---

## 15. LAYER DIAGRAM AND REQUEST FLOW

```
HTTP Request
     │
     ▼
┌─────────────────────────────────────────────┐
│  api/routes  (FastAPI route handlers)        │
│  Maps HTTP ↔ domain schemas                  │
│  Injects: ModelService, MetricsService       │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  services/model_service  (orchestration)     │
│  Lock acquisition, version resolution,       │
│  detector dispatch, TrainResponse mapping    │
│                                              │
│  services/validation_service (7-rule chain)  │
│  services/metrics_service (latency tracking) │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  repository/model_repository  (filesystem)   │
│  joblib.dump/load, metadata.json, index.json │
│  Atomic index writes via os.replace()        │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  domain/  (pure logic, no I/O)               │
│  models.py  — AnomalyDetectionModel,         │
│               IsolationForestDetector        │
│  schemas.py — TimeSeries, DataPoint,         │
│               TrainResponse, PredictionResp  │
│  exceptions.py — typed error hierarchy       │
└─────────────────────────────────────────────┘

Cross-cutting:
  utils/logging.py       — JsonFormatter, RequestIdFilter, ContextVar
  utils/concurrency.py   — LockManager (threading.Lock per series_id)
  api/middleware.py      — request_id generation and propagation
  api/error_handlers.py  — exception → normalized HTTP error mapping
```

**Read path** (`/predict`): route → `ModelService.predict()` → `ModelRepository.load()` → domain model → response. No lock acquired.

**Write path** (`/fit`): route → `ValidationService.validate()` → acquire series lock → `ModelService.train()` → `ModelRepository.save()` → release lock → response.

---

## 16. ARTIFACT SCHEMAS

**`metadata.json`** — written once per version, never mutated. `model_params` is a flat dict scoped to the detector of that version:

```json
{
  "version": "v1",
  "detector": "gaussian",
  "model_params": { "mean": float, "std": float },
  "n_samples": int,
  "trained_at": "ISO-8601Z",
  "training_duration_ms": float,
  "data_range": { "min_timestamp": int, "max_timestamp": int },
  "training_data": [ { "timestamp": int, "value": float } ]
}
```

`model_params` shape by detector:
- `gaussian`: `{ "mean": float, "std": float }`
- `isolation_forest`: `{ "n_estimators": int, "contamination": "auto", "score_threshold": float }`

`training_data` is always written; legacy models that predate this field return `PLOT_DATA_UNAVAILABLE`.

**`index.json`** — one per `(series_id, detector)`, updated atomically on each retrain:

```json
{
  "schema_version": "1",
  "series_id": "sensor_XYZ",
  "detector": "gaussian",
  "latest_version": "v2",
  "versions": ["v1", "v2"]
}
```

**Evolution rule:** fields may only be added, never renamed or removed, within the same storage layout. `schema_version` is present in `index.json` (currently `"1"`); a bump triggers a migration guard in `ModelRepository`. `metadata.json` does not carry `schema_version` — new fields are treated as optional at read time.

---

## 17. OPERATIONAL ERROR HANDLING

**Artifact corruption scenarios:**

| Scenario | Behavior | Recovery |
|---|---|---|
| `metadata.json` missing | `FileNotFoundError` raised; mapped to `404` on read/predict paths, or `422 INCOMPLETE_MODEL_METADATA` on strict list — see endpoint table below | Retrain the series |
| `model.joblib` corrupted | `joblib.load()` raises → uncaught `500` (full traceback logged) | Delete version dir, retrain |
| `index.json` missing | `get_index()` returns `None` → series invisible to reads | Retrain to recreate index |
| `index.json` partially written | Prevented by atomic `os.replace()` — partial write leaves `.tmp`, not `index.json` | No action needed |

**Retryable vs non-retryable:**

- **Retryable (transient):** `500 INTERNAL_ERROR` on read paths — safe to retry without side effects.
- **Not retryable:** `400` validation errors (data quality issue), `422 UNSUPPORTED_DETECTOR` (client fix required), `404 SERIES_NOT_FOUND` (requires retraining).

**Fail-fast vs graceful degrade by endpoint:**

| Endpoint | Policy |
|---|---|
| `POST /fit` | Fail-fast — validation rejects before any I/O or lock |
| `POST /predict` | Fail-fast — missing model → `404`, no fallback to other versions |
| `GET /models` (default) | Graceful degrade — skips series with missing metadata, returns remainder |
| `GET /models?strict=true` | Fail-fast — any incomplete metadata → `422` |
| `GET /models/{series_id}` | Fail-fast — unknown series → `404` |
| `GET /plot` | Fail-fast — missing `training_data` in metadata → `422` |

---

## 18. CONTRACT VERSIONING

**API versioning:** No version prefix in routes (implicit v1). Additive changes (new query params, new response fields) do not break existing callers. Breaking changes require a `/v2/` route prefix — not applied in this version.

**Artifact schema versioning:** `index.json` carries `schema_version` (currently `"1"`). `metadata.json` does not — new fields are treated as optional at read time, preserving backward compatibility with legacy models. A bump to `schema_version` in `index.json` is the migration trigger; any breaking change to `metadata.json` structure requires adding `schema_version` there first, along with a migration guard in `ModelRepository`.

**Field deprecation policy:** Fields are never removed in-place. Deprecated fields are kept readable but excluded from new writes. A migration script must be provided before any field removal.

**Detector namespace isolation:** Each `(series_id, detector)` pair is a fully independent storage namespace. Adding a new detector type requires no changes to existing artifacts or index files.

---

## 19. SLO / SLI TARGETS

Targets are defined for the current single-instance local deployment. Production targets with pre-loaded model cache are noted separately.

| Endpoint | p50 target | p99 target | Notes |
|---|---:|---:|---|
| `POST /predict` | < 10ms | < 50ms | Sequential load; with model pre-load: p99 < 5ms |
| `POST /fit` | < 500ms | < 2000ms | Dominated by training + `joblib.dump()` |
| `GET /models` | < 50ms | < 200ms | Filesystem metadata reads; scales with series count |
| `GET /models/{series_id}` | < 30ms | < 100ms | Single metadata read + quality derivation |
| `GET /plot` | < 200ms | < 500ms | matplotlib rendering; Agg backend, no display server |
| `GET /healthcheck` | < 5ms | < 20ms | In-memory only; no I/O |

Measurement: `X-Response-Time-Ms` response header is set by the metrics middleware on every request.

---

## 20. SECURITY

**Current scope (challenge deployment):**

- No authentication or authorization — all endpoints are publicly accessible.
- No rate limiting.
- No CORS policy — `CORSMiddleware` is not installed; without it FastAPI sends no CORS headers and browsers block cross-origin requests by default.
- `series_id` path parameter validated against filesystem-safe character set to prevent path traversal (`InvalidSeriesIdError` → `400`).
- No secrets in codebase or committed `.env` — `.env` is `.gitignore`d; `.env.example` contains only safe defaults.

**Accepted risks for this scope:** unauthenticated model training (any caller can overwrite a series), no audit trail beyond logs, single-instance with no network isolation.

**Production path:**

| Concern | Recommended control |
|---|---|
| Authentication | API key header or JWT via middleware |
| Rate limiting | nginx or API gateway upstream |
| CORS | Explicit allowlist in FastAPI `CORSMiddleware` |
| Audit trail | Structured logs (`LOG_FORMAT=json`) forwarded to SIEM |
| Network isolation | VPC/private subnet; API not exposed to public internet |

---

