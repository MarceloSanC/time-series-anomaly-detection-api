# SKILLS.MD — ANOMALY DETECTION API DEVELOPMENT GUIDE

## PROJECT CONTEXT

This is a production-oriented time series anomaly detection API built with FastAPI.
The system supports multiple `series_id`, model versioning, persistence, and real-time inference.
Development is time-constrained (5 stages) with AI-assisted coding (GPT-codex + Claude Sonnet 4.6).
The API contract for HTTP routes and payloads is defined by `docs/context/openapi_spec.yaml`.

Always read this file in full at the start of every coding session before writing any code.

---

## CORE PRINCIPLES

1. **Working software over perfect software.** Ship the simplest thing that works correctly, then improve.
2. **Explicit over implicit.** No magic. Every dependency is injected, every decision is visible in code.
3. **Fail loudly and clearly.** Every error must have a clear message, a correct HTTP status code, and a traceable log.
4. **Test the behavior, not the implementation.** Tests describe what the system does, not how.
5. **Configuration over hardcoding.** Every tunable parameter lives in `config.py` reading from `.env`.
6. **No premature optimization.** Optimize only after measuring, never before.
7. **MVP first, enhancements after.** No enhancement work is allowed before the MVP checklist is fully satisfied. No exceptions.

---

## ARCHITECTURE DECISIONS (DO NOT CHANGE WITHOUT EXPLICIT DISCUSSION)

- **Framework:** FastAPI with uvicorn
- **Persistence:** joblib on local filesystem, structured as `storage/{series_id}/{version}/`
- **ML library:** NumPy only (no sklearn, no torch, no extra ML dependencies)
- **Validation:** Pydantic v2 for schema validation, `ValidationService` for business rule validation
- **Concurrency:** `threading.Lock` per `series_id` managed by `LockManager`. Training is synchronous inside the lock. Do NOT use `asyncio.Lock` or `run_in_executor` in the MVP — it adds complexity without measurable benefit at this scale.
- **Versioning:** Incremental string versions (`v1`, `v2`, `v3`) with `index.json` per series. Write index atomically: write to temp file, then rename.
- **Metrics:** In-memory with `threading.Lock`. Track `request_count` and `total_time` per endpoint in the MVP. Calculate mean latency from these. Add `numpy.percentile` for p95/p99 only if the latency list size is bounded (cap at last 1000 samples).
- **Containerization:** Docker + docker-compose, single service
- **Logging:** Structured logging with `request_id` (correlation ID) injected via middleware into every log line.

---

## LAYER RESPONSIBILITIES

- **`domain/`** — Pure Python. No I/O, no HTTP, no filesystem. Only ML logic and Pydantic schemas.
- **`services/`** — Business logic. Calls repository and domain. Zero HTTP knowledge.
- **`repository/`** — All filesystem operations. Knows about storage paths. Zero business logic.
- **`api/`** — All HTTP concerns. Calls services only. Zero business logic, zero filesystem access.
- **`utils/`** — Shared infrastructure: logging setup, LockManager, concurrency helpers.

Any code that violates layer boundaries must be refactored immediately, not left as tech debt.

---

## PROJECT STRUCTURE

```
anomaly-detection-api/
├── app/
│   ├── main.py                   # FastAPI app, lifespan, routers, middleware
│   ├── config.py                 # Settings via pydantic-settings, reads .env
│   ├── dependencies.py           # FastAPI dependency injection
│   │
│   ├── domain/
│   │   ├── models.py             # AnomalyDetectionModel (core algorithm)
│   │   └── schemas.py            # Pydantic: DataPoint, TimeSeries, responses
│   │
│   ├── services/
│   │   ├── model_service.py      # Business logic: train, predict, list
│   │   ├── validation_service.py # Preflight validation rules
│   │   └── metrics_service.py    # In-memory metrics collection
│   │
│   ├── repository/
│   │   └── model_repository.py   # Persistence: save/load/list with joblib
│   │
│   ├── api/
│   │   ├── routes/
│   │   │   ├── fit.py
│   │   │   ├── predict.py
│   │   │   ├── plot.py
│   │   │   └── healthcheck.py
│   │   └── error_handlers.py     # Global exception handlers
│   │
│   └── utils/
│       ├── logging.py            # Structured logging + request_id middleware
│       └── concurrency.py        # LockManager: threading.Lock per series_id
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_model_service.py
│   │   ├── test_validation_service.py
│   │   └── test_model_repository.py
│   └── integration/
│       ├── test_fit_endpoint.py
│       ├── test_predict_endpoint.py
│       └── test_healthcheck_endpoint.py
│
├── scripts/
│   ├── benchmark.py              # Load test: 100 parallel inference requests
│   └── examples/
│       ├── fit_request.sh
│       └── predict_request.sh
│
├── storage/                      # Created at runtime, git-ignored
│   └── .gitkeep
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── Makefile
├── .env.example
├── docs/ai/skills.md             # This file
└── README.md
```

---

## INTERNAL DOMAIN SCHEMAS (source of truth for domain/services)

For API request/response payloads, follow `docs/context/openapi_spec.yaml` exactly.

API/domain mapping contract (must happen in `api/` layer):

- `Trainining.TrainData.timestamps` + `Trainining.TrainData.values` -> `TimeSeries.data[]`
- `Prediction.PredictData.timestamp` (OpenAPI `string`) -> internal `DataPoint.timestamp` (`int`)
- Internal `TrainResponse.n_samples` -> OpenAPI `Trainining.TrainResponse.points_used`
- Internal `PredictionResponse.is_anomaly` -> OpenAPI `Prediction.PredictResponse.anomaly`
- Internal `PredictionResponse.version` -> OpenAPI `Prediction.PredictResponse.model_version`

```python
# domain/schemas.py

from typing import Sequence, Optional
from pydantic import BaseModel, Field

class DataPoint(BaseModel):
    timestamp: int = Field(..., description="Unix timestamp")
    value: float = Field(..., description="Measured value")

class TimeSeries(BaseModel):
    data: Sequence[DataPoint] = Field(..., description="Ordered list of data points")

class TrainResponse(BaseModel):
    series_id: str
    version: str
    n_samples: int
    mean: float
    std: float
    training_duration_ms: float
    trained_at: str  # ISO 8601

class PredictionResponse(BaseModel):
    series_id: str
    version: str
    is_anomaly: bool
    value: float
    timestamp: int
    mean: float
    upper_bound: float  # mean + 3*std

class ModelInfo(BaseModel):
    series_id: str
    latest_version: str
    versions: list[str]
    trained_at: str
    n_samples: int

class ErrorResponse(BaseModel):
    error: str         # machine-readable error code, e.g. "SERIES_NOT_FOUND"
    message: str       # human-readable description
    detail: Optional[str] = None
    timestamp: str     # ISO 8601
```

---

## ANOMALY DETECTION MODEL

The model is provided by the challenge spec. Do NOT change the algorithm. Fix only the iteration bug silently:

```python
# domain/models.py

import numpy as np
from app.domain.schemas import DataPoint, TimeSeries

class AnomalyDetectionModel:
    mean: float
    std: float

    def fit(self, data: TimeSeries) -> "AnomalyDetectionModel":
        # FIXED: iterate over data.data (Sequence[DataPoint]), not data directly
        values = [d.value for d in data.data]
        self.mean = float(np.mean(values))
        self.std = float(np.std(values))
        return self

    def predict(self, data_point: DataPoint) -> bool:
        # Only detects positive anomalies (above mean + 3*std) — known limitation, documented in README
        return data_point.value > self.mean + 3 * self.std
```

Known limitation: the model only detects anomalies above the upper bound. It does NOT detect negative anomalies (below mean - 3*std). Document this in README under "Failure Modes". Do not change the algorithm.

---

## MODEL REPOSITORY

```python
# repository/model_repository.py

# Storage layout:
#   storage/{series_id}/index.json       ← list of versions + latest pointer
#   storage/{series_id}/{version}/model.joblib
#   storage/{series_id}/{version}/metadata.json

# index.json schema:
# {
#   "series_id": "sensor_XYZ",
#   "latest_version": "v3",
#   "versions": ["v1", "v2", "v3"]
# }

# metadata.json schema:
# {
#   "version": "v2",
#   "mean": 42.0,
#   "std": 3.5,
#   "n_samples": 150,
#   "trained_at": "2024-01-15T10:30:00Z",
#   "training_duration_ms": 12.4,
#   "data_range": {"min_timestamp": 1700000000, "max_timestamp": 1700086400}
# }

# CRITICAL: Write index.json atomically:
#   1. Write to index.json.tmp
#   2. os.replace(tmp_path, index_path)  ← atomic on POSIX systems
```

---

## VALIDATION SERVICE RULES

All rules must be configurable via `.env` / `config.py`. Defaults shown below.

```
MIN_DATA_POINTS = 30          # Reject series with fewer points — statistically unreliable
STD_THRESHOLD = 1e-10         # Reject constant series (std below this value)
```

Rules to enforce (in this order, fail fast):

1. **Minimum points:** `len(data.data) >= MIN_DATA_POINTS` → `400 INSUFFICIENT_DATA`
2. **Constant series:** `np.std(values) >= STD_THRESHOLD` → `400 CONSTANT_SERIES`
3. **Duplicate timestamps:** no repeated timestamp values → `400 DUPLICATE_TIMESTAMPS`
4. **Out of order:** timestamps must be strictly increasing → `400 UNORDERED_TIMESTAMPS`
5. **Non-finite values:** no NaN or Inf in values → `400 INVALID_VALUES`

Each rule must have at least one positive and one negative test in `test_validation_service.py`.

---

## CONCURRENCY RULES

```python
# utils/concurrency.py

import threading
from collections import defaultdict

class LockManager:
    """Per-series_id threading.Lock. Serializes writes to the same series."""
    
    def __init__(self):
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._meta_lock = threading.Lock()  # protects _locks dict itself

    def get_lock(self, series_id: str) -> threading.Lock:
        with self._meta_lock:
            return self._locks[series_id]
```

Usage in `model_service.py`:

```python
lock = self.lock_manager.get_lock(series_id)
with lock:
    # validate → train → save → update index
```

This allows concurrent training of different series while serializing training of the same series.

---

## METRICS SERVICE

MVP implementation — keep it simple and correct:

```python
# services/metrics_service.py

import threading
import time
from collections import defaultdict

class MetricsService:
    def __init__(self, max_latency_samples: int = 1000):
        self._lock = threading.Lock()
        self._max_samples = max_latency_samples
        self._counts: dict[str, int] = defaultdict(int)
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._errors: dict[str, int] = defaultdict(int)
        self._start_time = time.time()

    def record(self, endpoint: str, latency_ms: float, error: bool = False):
        with self._lock:
            self._counts[endpoint] += 1
            samples = self._latencies[endpoint]
            if len(samples) >= self._max_samples:
                samples.pop(0)  # evict oldest
            samples.append(latency_ms)
            if error:
                self._errors[endpoint] += 1

    def snapshot(self) -> dict:
        import numpy as np
        with self._lock:
            result = {}
            for endpoint, latencies in self._latencies.items():
                arr = np.array(latencies)
                result[endpoint] = {
                    "request_count": self._counts[endpoint],
                    "error_count": self._errors[endpoint],
                    "mean_latency_ms": float(np.mean(arr)) if len(arr) else 0.0,
                    "p95_latency_ms": float(np.percentile(arr, 95)) if len(arr) else 0.0,
                    "p99_latency_ms": float(np.percentile(arr, 99)) if len(arr) else 0.0,
                }
            return {
                "uptime_seconds": time.time() - self._start_time,
                "endpoints": result,
            }
```

---

## VERSIONING STRATEGY

- Versions are strings: `v1`, `v2`, `v3` — never timestamps as primary version identifier.
- `index.json` per series tracks `latest_version` and ordered list of `versions`.
- `POST /predict/{series_id}` → defaults to `latest_version` when `version` query param is omitted.
- `POST /predict/{series_id}?version=v2` → loads specific version.
- `index.json` must be written atomically (write-then-rename pattern).
- Next version number is derived from `len(existing_versions) + 1`.

---

## API ERROR CONTRACT

All errors must return a JSON body matching `ErrorResponse`. Never expose raw Python exceptions.

| HTTP Status | error code              | When to use                                  |
|-------------|-------------------------|----------------------------------------------|
| 400         | INSUFFICIENT_DATA       | Series has fewer than MIN_DATA_POINTS points |
| 400         | CONSTANT_SERIES         | Series std is effectively zero               |
| 400         | DUPLICATE_TIMESTAMPS    | Repeated timestamps in input                 |
| 400         | UNORDERED_TIMESTAMPS    | Timestamps not strictly increasing           |
| 400         | INVALID_VALUES          | NaN or Inf values in series                  |
| 404         | SERIES_NOT_FOUND        | series_id does not exist in storage          |
| 404         | VERSION_NOT_FOUND       | Requested version does not exist             |
| 409         | TRAINING_CONFLICT       | Concurrent training of same series (rare)    |
| 422         | (Pydantic default)      | Schema validation failure                    |
| 500         | INTERNAL_ERROR          | Unexpected server error                      |

---

## LOGGING STANDARD

Every log line must include `request_id` (correlation ID). Set it in middleware, store in `contextvars.ContextVar`.

```python
import logging
import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="none")

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get("none")
        return True
```

Log format: `%(asctime)s [%(request_id)s] %(levelname)s %(name)s: %(message)s`

Log levels:
- `INFO` — request start/end, train completed, predict result
- `WARNING` — validation rejection, version not found
- `ERROR` — unexpected exceptions

---

## MODEL METADATA ENRICHMENT

Every trained model's `metadata.json` must include:

```json
{
  "version": "v2",
  "mean": 42.0,
  "std": 3.5,
  "n_samples": 150,
  "trained_at": "2024-01-15T10:30:00Z",
  "training_duration_ms": 12.4,
  "data_range": {
    "min_timestamp": 1700000000,
    "max_timestamp": 1700086400,
    "min_value": 10.2,
    "max_value": 98.7
  }
}
```

This enables model inspection without loading the joblib file and powers the `/plot` endpoint.

---

## VISUALIZATION ENDPOINT

`GET /plot?series_id=sensor_XYZ&version=v3`

Implementation rules:
- Use `matplotlib` with `matplotlib.use('Agg')` at import time — required for headless Docker.
- Load training metadata from `metadata.json` (do NOT re-load the model just for plotting).
- Plot: scatter of training values over timestamps, horizontal line at `mean`, two dashed lines at `mean ± 3*std`.
- Return as `StreamingResponse` with `media_type="image/png"`.
- Keep visual simple — no custom fonts, no elaborate styling. Functional only.
- Only implement after MVP checklist is 100% complete.

---

## PERFORMANCE TESTING

Script: `scripts/benchmark.py`

Requirements:
- Send 100 parallel inference requests using `asyncio` + `httpx.AsyncClient`.
- Target: `POST /predict/{series_id}` on a pre-trained series.
- Measure: total time, individual latencies, p50/p95/p99.
- Output: print summary + save results to `scripts/benchmark_results.json`.
- Include results in README.

---

## CONFIGURATION (.env.example)

```env
# App
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO

# Storage
STORAGE_PATH=./storage

# Validation thresholds (configurable)
MIN_DATA_POINTS=30
STD_THRESHOLD=1e-10

# Metrics
MAX_LATENCY_SAMPLES=1000
```

---

## TESTING RULES

- Unit tests: no filesystem, no HTTP. Use `tmp_path` pytest fixture for any needed I/O. Mock repository in service tests.
- Integration tests: use FastAPI `TestClient` (synchronous) with a `tmp_path` storage directory injected via dependency override.
- Every `ValidationService` rule → at least 1 passing case + 1 rejection case.
- Every API endpoint → at least 1 happy path + 1 error path.
- Run with: `pytest -v --tb=short`

---

## MVP CHECKLIST (must be 100% complete before any enhancement)

- [ ] `POST /fit/{series_id}` — trains and persists model, returns OpenAPI-compatible response
- [ ] `POST /predict/{series_id}` — returns prediction with version (defaults to latest)
- [ ] `GET /healthcheck` — returns 200 with OpenAPI-compatible payload
- [ ] Versioning: retraining same series_id creates new version, does not overwrite old
- [ ] Persistence: models survive container restart
- [ ] Docker: `docker-compose up` works from zero with no manual steps
- [ ] Tests: all unit + integration tests pass with `pytest -v`
- [ ] No raw Python exceptions exposed to API clients

---

## ENHANCEMENT CHECKLIST (only after MVP is complete)

- [ ] Preflight validation (all 5 rules implemented and tested)
- [ ] `POST /predict/{series_id}?version=v2` — predict with specific version
- [ ] `GET /plot?series_id=X&version=v3` — visualization endpoint
- [ ] `scripts/benchmark.py` — load test + results saved to JSON
- [ ] Benchmark results documented in README
- [ ] `request_id` correlation ID in all log lines
- [ ] Failure modes documented in README

---

## CODE STYLE

- Python 3.11+
- Type hints on **all** function signatures — no exceptions
- Pydantic v2 models for all request/response objects
- No global mutable state except `LockManager` and `MetricsService` (both explicitly initialized in lifespan)
- Docstrings on public methods only — brief and functional, not encyclopedic
- Error handling: raise domain exceptions in services, catch and translate in API layer
- Never use bare `except:` — always catch specific exception types

---

## KNOWN LIMITATIONS (document these in README, do not fix)

1. **One-sided anomaly detection:** The model only flags values above `mean + 3*std`. Values below `mean - 3*std` are not flagged. This is a property of the provided algorithm, not a system bug.
2. **In-memory metrics:** Metrics are lost on restart. Not a concern for this scope.
3. **Local filesystem storage:** Not suitable for horizontally scaled deployments. Acceptable for single-instance use.
4. **Constant series rejection:** A series with extremely low variance may be legitimate in some domains. The `STD_THRESHOLD` is configurable for this reason.

---

## HOW TO USE THIS FILE WITH LLMs

When starting a session with GPT-codex or Claude:

1. Paste this entire file as system context before any code request.
2. Specify which layer you are working on (domain / services / repository / api / utils / tests).
3. Specify what already exists (avoid regenerating completed files).
4. For architecture decisions: ask Claude. For code generation: ask GPT-codex.
5. After any generated code, verify against this checklist:
   - [ ] Type hints on all signatures?
   - [ ] Error handling present?
   - [ ] Correct layer (no boundary violations)?
   - [ ] Logging included?
   - [ ] No hardcoded values that should be in config?

---

## WHAT TO NEVER DO

- Never use `pickle` directly — always use `joblib`
- Never put business logic in route handlers
- Never access filesystem from `api/` layer
- Never swallow exceptions silently — always log and re-raise or translate
- Never hardcode paths — always derive from `config.STORAGE_PATH`
- Never block the event loop with long synchronous operations (training uses threading.Lock, not async)
- Never return raw Python exceptions to API clients
- Never start enhancement work before MVP checklist is 100% satisfied
- Never write `index.json` without the atomic write-then-rename pattern
- Never expose `series_id` directory traversal (sanitize inputs)
