# Time Series Anomaly Detection API

A production-oriented REST API for univariate time-series anomaly detection, built with FastAPI.

The service:
- trains models per `series_id`
- versions each retrain (`v1`, `v2`, ...)
- persists model artifacts on local filesystem (`storage/`)
- serves real-time predictions
- exposes health metrics and visualization endpoints

## Capabilities

| Area | What's built |
|---|---|
| **Detectors** | Gaussian (parametric, 3σ threshold) and Isolation Forest — switchable via `?detector=` on all endpoints |
| **Versioning** | Incremental version labels (`v1`, `v2`, …) scoped per `(series_id, detector)` with full lineage tracking |
| **Storage** | Detector-scoped artifact layout: `storage/{series_id}/{detector}/{version}/` |
| **Validation** | Fail-fast pipeline with 7 configurable rules; industrial sensor rules (flat-line, temporal-gap) opt-in |
| **Introspection** | `/models*` endpoints expose model inventory, data quality indicators, and version metadata |
| **Logging** | `LOG_FORMAT=text\|json`; JSON mode emits structured lines with `request_id` and ML event fields |
| **Observability** | In-memory latency percentiles (p50/p95/p99) per endpoint; visualization via `/plot` |
| **Analysis scripts** | Benchmark (latency under load), detector comparison (TPR/FPR), drift analysis (static vs rolling) |

## Architecture Overview

**Layered design:** `api` → `services` → `repository` → `domain`. The API layer maps HTTP contracts to domain schemas; `ModelService` orchestrates training and inference; `ModelRepository` handles all filesystem I/O; domain models (`AnomalyDetectionModel`, `IsolationForestDetector`) are transport-agnostic.

**Concurrency:** Training of different `series_id` values runs in parallel. Training of the same `series_id` is serialized with a `threading.Lock` to guarantee correct version ordering. Inference is stateless and fully concurrent.

**Validation pipeline:** `ValidationService` runs seven fail-fast rules before the training lock is acquired — structural checks (min points, constant series, NaN/inf, duplicate/unordered timestamps) plus optional industrial sensor rules (flat-line, temporal-gap).

**Request correlation:** A UUID `request_id` is generated per request in middleware, stored in a `ContextVar`, and injected into every log record — enabling full request traceability across all log lines.

For detailed decisions and trade-offs, see [Architecture](docs/project/ARCHITECTURE.md).

## Requirements

Runtime:
- Docker
- Docker Compose

Local development/test:
- Python 3.11+
- `venv` support (`python3-venv` on Debian/Ubuntu)

## Setup

```bash
git clone https://github.com/MarceloSanC/time-series-anomaly-detection-api.git
cd time-series-anomaly-detection-api
cp .env.example .env
docker compose up --build -d
curl http://localhost:8000/healthcheck
```

Verified from-scratch test flow (clone -> up -> test):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
```

Coverage quality gate:
- minimum enforced coverage: `80%` (`--cov-fail-under=80`)
- generate terminal + HTML report with:

```bash
make coverage
```

Optional container-only validation:

```bash
docker compose up --build -d
docker compose logs -f api
```

## Common Make Targets

| Target | Command | Description |
|---|---|---|
| `make install` | `pip install -e ".[dev]"` | Install all dependencies including dev tools |
| `make test` | `pytest -v` | Run full test suite with coverage gate |
| `make coverage` | `pytest --cov=app --cov-report=html` | Generate HTML coverage report at `htmlcov/index.html` |
| `make lint` | `ruff check app tests` | Run Ruff linter (rules E and F) |
| `make check` | `make lint && make test` | Pre-PR gate: lint then test |
| `make run` | `uvicorn app.main:app --reload` | Start local dev server on port 8000 |
| `make docker-up` | `docker compose up -d` | Start service in background |
| `make docker-down` | `docker compose down -v` | Stop service and remove volumes |
| `make docker-test` | `docker compose run --rm api-tests` | Run tests inside container |
| `make benchmark` | `python scripts/benchmark.py` | Run 100-request parallel inference benchmark |
| `make smoke` | `./scripts/manual/stage2_smoke_test.sh` | Manual smoke test against running service |

## Endpoint Examples

Train:

Script (detector default `gaussian`):

```bash
./scripts/examples/fit_request.sh
```

Script (explicit `isolation_forest` detector):

```bash
DETECTOR=isolation_forest ./scripts/examples/fit_request.sh
```

Equivalent `curl`:

```bash
curl --fail-with-body -sS -X POST "http://localhost:8000/fit/sensor_XYZ" \
  -H "Content-Type: application/json" \
  --data-binary @- <<'JSON'
{
  "timestamps": [
    1700000001, 1700000002, 1700000003, 1700000004, 1700000005,
    1700000006, 1700000007, 1700000008, 1700000009, 1700000010,
    1700000011, 1700000012, 1700000013, 1700000014, 1700000015,
    1700000016, 1700000017, 1700000018, 1700000019, 1700000020,
    1700000021, 1700000022, 1700000023, 1700000024, 1700000025,
    1700000026, 1700000027, 1700000028, 1700000029, 1700000030
  ],
  "values": [
    10.0, 10.3, 10.6, 10.9, 11.2,
    11.5, 11.8, 12.0, 12.3, 12.6,
    12.9, 13.2, 13.5, 13.8, 14.0,
    14.3, 14.6, 14.9, 15.2, 15.5,
    15.8, 16.1, 16.4, 16.7, 17.0,
    17.3, 17.6, 17.9, 18.2, 18.5
  ]
}
JSON
```

Predict:

```bash
curl --fail-with-body -sS -X POST "http://localhost:8000/predict/sensor_XYZ" \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"1700000100","value":99.0}'

# explicit version and detector:
curl --fail-with-body -sS -X POST "http://localhost:8000/predict/sensor_XYZ?version=v1&detector=isolation_forest" \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"1700000100","value":99.0}'
```

Healthcheck:

```bash
curl --fail-with-body -sS http://localhost:8000/healthcheck
```

Model introspection (additive extension):

```bash
# List tracked series (default tolerant mode)
curl --fail-with-body -sS "http://localhost:8000/models"

# Strict mode: fail-fast if any latest metadata is incomplete
curl --fail-with-body -sS "http://localhost:8000/models?strict=true"

# Detector-scoped list
curl --fail-with-body -sS "http://localhost:8000/models?detector=isolation_forest"

# Series detail with derived data_quality
curl --fail-with-body -sS "http://localhost:8000/models/sensor_XYZ"
curl --fail-with-body -sS "http://localhost:8000/models/sensor_XYZ?detector=isolation_forest"

# Version metadata summary (training_data excluded by default)
curl --fail-with-body -sS "http://localhost:8000/models/sensor_XYZ/versions/v1"
curl --fail-with-body -sS "http://localhost:8000/models/sensor_XYZ/versions/v1?detector=isolation_forest"

# Version metadata including persisted training_data
curl --fail-with-body -sS "http://localhost:8000/models/sensor_XYZ/versions/v1?include_data=true"
```

These `/models*` endpoints are additive introspection extensions and do not change
the core OpenAPI-defined contracts for `/fit`, `/predict`, and `/healthcheck`.

Visualization:

```bash
curl "http://localhost:8000/plot?series_id=sensor_XYZ" --output plot.png
curl "http://localhost:8000/plot?series_id=sensor_XYZ&version=v1" --output plot_v1.png
```

Example output:

![Plot endpoint example showing training points, mean line, and +/-3 sigma bounds](docs/assets/plot_example.png)

The image shows training points (scatter), the model mean, and upper/lower 3-sigma thresholds.

## Analysis Scripts

Three standalone scripts cover latency benchmarking, detector quality comparison, and concept drift analysis. All output JSON artifacts to `scripts/` and require the service running at `http://localhost:8000`.

---

### Benchmark — inference latency under parallel load

Measures p50/p95/p99 latency and throughput under 100 concurrent prediction requests.

```bash
make benchmark                                        # gaussian default
python scripts/benchmark.py --detector isolation_forest
python scripts/benchmark.py --both-detectors          # saves combined JSON
```

Output saved to `scripts/benchmark_results.json`.

Latest recorded run (`--both-detectors`):

| Metric | Gaussian | Isolation Forest |
|---|---:|---:|
| p50 (ms) | 210.95 | 3310.11 |
| p95 (ms) | 376.48 | 4280.77 |
| p99 (ms) | 381.35 | 4309.49 |
| avg (ms) | 233.09 | 3085.88 |
| min (ms) | 24.77 | 63.32 |
| max (ms) | 381.46 | 4312.52 |
| throughput (req/s) | 249.62 | 23.09 |
| total duration (s) | 0.40 | 4.33 |

- Gaussian p50 (~211ms) under 100 parallel requests is dominated by `joblib.load()` I/O contention — the O(1) prediction cost is negligible at this scale.
- Isolation Forest p50 (~3310ms) compounds I/O contention with GIL pressure from scikit-learn's `score_samples()` traversing 100 decision trees per call — ~16× slower throughput than gaussian under load.
- Results are environment-sensitive; compare trends across repeated runs rather than absolute values.

---

### Detector Comparison — detection quality on synthetic anomalies

Trains both detectors via the API on the same gaussian-distributed dataset (150 points, mean=50, std=5), then evaluates on a held-out test split with 20 positive spike anomalies injected at mean+4.5*std. Models are persisted after the run.

```bash
.venv/bin/python scripts/compare_detectors.py
```

Output saved to `scripts/detector_comparison.json`. Models are persisted in storage after the run. The plot below shows the **training data only** (150 normal points) with the fitted gaussian bounds — anomalies are injected exclusively into the held-out test split and are not visible here:

```bash
curl "http://localhost:8000/plot?series_id=compare_gaussian" --output compare_plot.png
```

![Gaussian training data (150 normal points) with fitted mean and ±3σ bounds — test anomalies not shown](docs/assets/compare_plot.png)

Latest recorded run:

| Metric | Gaussian | Isolation Forest |
|---|---:|---:|
| TPR | 100.00% | 100.00% |
| FPR | 0.00% | 17.50% |
| p50 latency (ms) | 3.80 | 31.98 |
| p95 latency (ms) | 7.20 | 38.86 |
| p99 latency (ms) | 9.04 | 81.89 |

- Both detectors achieved 100% TPR on anomalies injected at mean+4.5*std.
- Gaussian achieved 0% FPR; IsolationForest flagged 17.5% of normal points (higher FPR by design of its 10th-percentile threshold).
- This result is expected on gaussian-distributed data — the gaussian detector is a parametric fit to the exact distribution. IsolationForest holds the advantage on non-gaussian, multimodal, or clustered-anomaly scenarios, and is the only option for detecting negative outliers.
- Under sequential low-load conditions, the latency gap (p50: ~4ms gaussian vs ~32ms isolation_forest) reflects the algorithmic cost difference: gaussian predict is O(1) arithmetic; isolation_forest traverses 100 decision trees per call via scikit-learn `score_samples`. Under high concurrency this gap widens further — GIL contention from tree traversal serializes isolation_forest predictions, driving throughput down to ~23 req/s vs ~250 req/s for gaussian (see benchmark results above).

---

### Drift Analysis — concept drift behavior over time

Simulates 6 hours of sensor data with gradual mean drift starting at hour 1, and compares a static threshold (trained on healthy baseline) against a rolling-window adaptive threshold.

```bash
.venv/bin/python scripts/drift_analysis.py
```

Output saved to `scripts/drift_analysis_results.json`. Three plots are generated under `docs/assets/`.

![Time series with drift and anomaly markers](docs/assets/drift_analysis_timeseries.png)

![Hourly FPR: baseline vs rolling window](docs/assets/drift_analysis_fpr_hourly.png)

Latest recorded run:

| Metric | Static Baseline | Rolling Window |
|---|---:|---:|
| FPR (overall) | 52.32% | 0.019% |
| Alerts/hour | 1892.8 | 20.3 |
| Time to instability | 3h | — (stable) |

- The static threshold becomes unreliable after drift begins (~hour 1), saturating at 52% FPR by hour 3.
- The rolling window adapts continuously, keeping FPR near zero throughout the drift window.
- FPR reduction: ~280,875× relative to static baseline.

## Known Limitations

All limitations below are deliberate design decisions or known trade-offs with documented evolution paths.

**Algorithm — Gaussian detector**

| Limitation | Notes | Reference |
|---|---|---|
| One-sided detection only | Values below `mean − 3σ` are never flagged; use `isolation_forest` for symmetric detection | [Modeling Notes §1](docs/project/MODEL_DESIGN_NOTES.md) |
| Sensitive to training outliers | `mean`/`std` are non-robust; a single spike in training data widens thresholds and reduces sensitivity | [Modeling Notes §1](docs/project/MODEL_DESIGN_NOTES.md) |
| Stationarity assumption | Thresholds are static; performance degrades under concept drift without retraining | [Modeling Notes §1](docs/project/MODEL_DESIGN_NOTES.md) |
| No contextual detection | One global threshold — same value can be normal in one regime and anomalous in another | [Modeling Notes §1](docs/project/MODEL_DESIGN_NOTES.md) |

**Algorithm — Isolation Forest detector**

| Limitation | Notes | Reference |
|---|---|---|
| High FPR on gaussian-distributed data | 10th-percentile threshold flags ~10% of normal points by design; use gaussian for parametric distributions | [V2 Roadmap Stage D](docs/project/V2_ROADMAP.md) |
| Masking effect on low-variance data | Insufficient training variance causes all points to score similarly, degrading detection | [V2 Roadmap Stage D](docs/project/V2_ROADMAP.md) |

**System**

| Limitation | Notes | Reference |
|---|---|---|
| Single-instance persistence | Filesystem layout not shared across instances; production path: S3/GCS via same `ModelRepository` interface | [Architecture §1, §13](docs/project/ARCHITECTURE.md) |
| Metrics reset on restart | In-memory only; production path: Prometheus + Grafana | [Architecture §4](docs/project/ARCHITECTURE.md) |
| Inference latency under load | `joblib.load()` on critical path; production path: pre-load latest model into `app.state` on startup (expected p99 < 5ms) | [Architecture §13](docs/project/ARCHITECTURE.md) |
| Training blocks series lock | Training of the same `series_id` is serialized; acceptable at O(n) model size | [Architecture §2](docs/project/ARCHITECTURE.md) |



## Troubleshooting

**API error reference**

All errors follow the same normalized shape: `{"error": "CODE", "message": "...", "detail": null, "timestamp": "..."}`.

`400 Bad Request` — training data rejected by validation pipeline:

| Error code | Cause |
|---|---|
| `INSUFFICIENT_DATA` | Fewer than `MIN_DATA_POINTS` points (default: 30) |
| `CONSTANT_SERIES` | Series standard deviation is below `STD_THRESHOLD` |
| `DUPLICATE_TIMESTAMPS` | Two or more points share the same timestamp |
| `UNORDERED_TIMESTAMPS` | Timestamps are not strictly increasing |
| `INVALID_VALUES` | Series contains `NaN` or infinite values |
| `FLAT_LINE_DETECTED` | Trailing window is constant (only when `FLAT_LINE_ENABLED=true`) |
| `TEMPORAL_GAP_DETECTED` | Max interval exceeds `MAX_TEMPORAL_GAP_FACTOR × median(intervals)` (only when `TEMPORAL_GAP_ENABLED=true`) |
| `INVALID_SERIES_ID` | `series_id` is empty or contains unsafe characters for filesystem storage |

`404 Not Found`:

| Error code | Cause |
|---|---|
| `SERIES_NOT_FOUND` | `series_id` does not exist in storage |
| `VERSION_NOT_FOUND_FOR_DETECTOR` | Requested `version` exists for a different detector or not at all |

`422 Unprocessable Entity`:

| Error code | Cause |
|---|---|
| `UNSUPPORTED_DETECTOR` | `?detector=` value is not `gaussian` or `isolation_forest` |
| `PLOT_DATA_UNAVAILABLE` | Model was trained before `training_data` was persisted — retrain to fix |
| `INCOMPLETE_MODEL_METADATA` | Metadata missing for a series version when `?strict=true` on `/models` |
| `VALIDATION_ERROR` | Request payload failed schema validation (missing fields, wrong types) |

**Docker env_file override does not work with shell-prefixed variables**

Running `FLAT_LINE_ENABLED=true docker compose up` does NOT enable the flag inside the container — `env_file` values in `docker-compose.yml` are passed directly to the container and are not overridden by shell-prefixed variables. Edit `.env` directly and restart with `--force-recreate`:

```bash
# Edit .env, then:
docker compose up --build --force-recreate
```

**Quick Docker validation**

```bash
# Start and verify the service is healthy
make docker-up
curl http://localhost:8000/healthcheck

# Run containerized test suite
make docker-test

# Inspect logs
docker compose logs -f api

# Tear down
make docker-down
```

## Architecture Decisions (Brief)

Key decisions and their rationale are documented in [Architecture](docs/project/ARCHITECTURE.md). Summary:

| Decision | Choice | Reference |
|---|---|---|
| Model persistence | `joblib` files under `storage/{series_id}/{detector}/{version}/`; metadata in separate `metadata.json` for inspection without deserialization | [§1](docs/project/ARCHITECTURE.md) |
| Concurrency | `threading.Lock` per `series_id` — serializes retraining of same series, parallel across different series; `asyncio.Lock` rejected (numpy/joblib not async-safe) | [§2](docs/project/ARCHITECTURE.md) |
| Versioning | Incremental string labels (`v1`, `v2`, …) per `(series_id, detector)`; `index.json` written atomically via rename to prevent corruption | [§3](docs/project/ARCHITECTURE.md) |
| Validation | Fail-fast pipeline runs before lock acquisition; thresholds injected via constructor for test isolation | [§5](docs/project/ARCHITECTURE.md) |
| API error contract | Normalized `{"error", "message", "detail", "timestamp"}` shape for all errors; status codes follow OpenAPI spec | [§6](docs/project/ARCHITECTURE.md) |
| Validation extensions | Flat-line and temporal-gap rules are opt-in via flags — always-on would reject valid non-industrial data; new rules require only a typed exception + one error map entry | [§11](docs/project/ARCHITECTURE.md) |
| Multi-detector contract | `?detector=` is additive on all endpoints; omitting it resolves to `gaussian` namespace — backward-compatible | [§14](docs/project/ARCHITECTURE.md) |

## Configuration

Copy `.env.example` to `.env` and adjust as needed.
`.env.example` is a documented baseline with safe defaults for local execution and Docker.
Use `.env` for real runtime values and keep it untracked.

| Variable | Default | Description |
|---|---|---|
| `APP_PORT` | `8000` | API server port |
| `LOG_LEVEL` | `INFO` | Python log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_FORMAT` | `text` | Log output format — `text` (human-readable) or `json` (structured lines) |
| `STORAGE_PATH` | `./storage` | Model artifact directory |
| `MIN_DATA_POINTS` | `30` | Minimum points required for training |
| `STD_THRESHOLD` | `1e-10` | Minimum std threshold for constant-series rejection |
| `MAX_LATENCY_SAMPLES` | `1000` | Sliding window size for latency percentiles |
| `FLAT_LINE_ENABLED` | `false` | Enable flat-line detection rule (opt-in; targets frozen/disconnected sensors) |
| `FLAT_LINE_WINDOW` | `10` | Number of trailing points checked for flat-line condition |
| `TEMPORAL_GAP_ENABLED` | `false` | Enable temporal-gap detection rule (opt-in; targets sampling instability) |
| `MAX_TEMPORAL_GAP_FACTOR` | `2.0` | Maximum interval factor — rejects if `max(intervals) > factor × median(intervals)` |

## Structured Logging

The service supports two log output formats, controlled by the `LOG_FORMAT` environment variable.

| Value | Behaviour |
|---|---|
| `text` (default) | Human-readable lines — `timestamp [request_id] LEVEL logger: message` |
| `json` | One JSON object per line, suitable for log aggregation platforms (Datadog, Loki, CloudWatch) |

Switch format at startup:

```bash
# local development (text, default)
LOG_FORMAT=text make run

# structured JSON output
LOG_FORMAT=json .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Sample JSON log line emitted after a training request:

```json
{
  "timestamp": "2026-04-15 18:00:00,123",
  "level": "INFO",
  "logger": "app.services.model_service",
  "message": "Training completed",
  "request_id": "4a3f1b2e-...",
  "event": "model_trained",
  "series_id": "sensor_XYZ",
  "detector": "gaussian",
  "version": "v1",
  "n_samples": 150,
  "duration_ms": 4.21,
  "mean": 50.03,
  "std": 4.97
}
```

Recommended fields for filtering in external log platforms:

| Field | Use case |
|---|---|
| `event` | Filter by operation type (`model_trained`, `prediction_served`) |
| `series_id` | Trace all activity for a specific sensor/series |
| `request_id` | Correlate all log lines for a single HTTP request |
| `is_anomaly` | Alert on prediction decisions |
| `detector` | Segment metrics by detector family |
| `version` | Identify which model version served a prediction |

## Documentation

- [Docs Index](docs/README.md)
- [V1 Roadmap](docs/project/V1_ROADMAP.md)
- [V2 Roadmap](docs/project/V2_ROADMAP.md)
- [Architecture](docs/project/ARCHITECTURE.md)
- [Modeling Notes](docs/project/MODEL_DESIGN_NOTES.md)
- [AI Usage](docs/ai/LLM_USAGE.md)
- [Git Protocol](docs/process/GIT_PROTOCOL.md)
- API live docs: `/docs` (Swagger UI) and `/redoc` (ReDoc)
