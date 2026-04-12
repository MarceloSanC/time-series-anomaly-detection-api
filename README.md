# time-series-anomaly-detection-api

A production-oriented REST API for univariate time-series anomaly detection, built with FastAPI.

The service:
- trains models per `series_id`
- versions each retrain (`v1`, `v2`, ...)
- persists model artifacts on local filesystem (`storage/`)
- serves real-time predictions
- exposes health metrics and visualization endpoints

## Architecture Overview

Each `series_id` keeps an independent model lineage. Retraining creates a new version without overwriting previous versions. Predictions default to latest version, but support explicit historical versions via `?version=`.

Concurrent training is parallel across different `series_id` values and serialized for the same `series_id` using per-series locking.

For architecture decisions and trade-offs, see [Architecture](docs/project/ARCHITECTURE.md).

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

Optional container-only validation:

```bash
docker compose up --build -d
docker compose logs -f api
```

## Endpoint Examples

Train:

```bash
./scripts/examples/fit_request.sh
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
./scripts/examples/predict_request.sh
VERSION_QUERY=v1 ./scripts/examples/predict_request.sh
curl --fail-with-body -sS -X POST "http://localhost:8000/predict/sensor_XYZ?version=v1" \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"1700000100","value":99.0}'
```

Healthcheck:

```bash
curl --fail-with-body -sS http://localhost:8000/healthcheck
```

Visualization:

```bash
curl "http://localhost:8000/plot?series_id=sensor_XYZ" --output plot.png
curl "http://localhost:8000/plot?series_id=sensor_XYZ&version=v1" --output plot_v1.png
```

Example output:

![Plot endpoint example showing training points, mean line, and +/-3 sigma bounds](docs/assets/plot_example.png)

The image shows training points (scatter), the model mean, and upper/lower 3-sigma thresholds.

## Benchmark

Run Stage 4 benchmark (100 parallel inference requests):

```bash
.venv/bin/python scripts/benchmark.py
```

Benchmark output is saved to `scripts/benchmark_results.json`.

Latest recorded run:

| Metric | Value |
|---|---:|
| p50 (ms) | 211.26 |
| p95 (ms) | 279.84 |
| p99 (ms) | 282.96 |
| avg (ms) | 204.03 |
| min (ms) | 19.92 |
| max (ms) | 283.93 |
| throughput (req/s) | 336.42 |
| total duration (s) | 0.30 |

Interpretation:
- `min` usually reflects early requests that reached a less busy server state.
- `p99` reflects tail latency under burst concurrency and queueing.
- With 100 simultaneous requests on a local single-process server, this spread is expected.

## Known Limitations

- The baseline algorithm detects only positive anomalies (`value > mean + 3*std`).
- Negative outliers are not flagged by design. Example: if `mean=100` and `std=5`, a value of `50`
  (i.e. `mean - 10*std`) still returns `anomaly=false` in `/predict`.
- Metrics are in-memory and reset on service restart.
- Persistence is local filesystem (`storage/`), suitable for single-instance deployments.
- `/plot` requires metadata that includes `training_data`; legacy models without it return `422 PLOT_DATA_UNAVAILABLE`.

## Architecture Decisions (Brief)

- Persistence uses local filesystem artifacts (`joblib` + `metadata.json`) under `storage/{series_id}/{version}`.
- Concurrency uses per-series locks (`threading.Lock`) to serialize retraining of the same `series_id`.
- Validation is fail-fast and runs before training lock acquisition.
- API contract follows `docs/context/openapi_spec.yaml`; internal schemas are mapped in the API layer.
- Metrics are in-memory with bounded latency windows for percentile calculations.

## Configuration

Copy `.env.example` to `.env` and adjust as needed.
`.env.example` is a documented baseline with safe defaults for local execution and Docker.
Use `.env` for real runtime values and keep it untracked.

| Variable | Default | Description |
|---|---|---|
| `APP_PORT` | `8000` | API server port |
| `STORAGE_PATH` | `./storage` | Model artifact directory |
| `MIN_DATA_POINTS` | `30` | Minimum points required for training |
| `STD_THRESHOLD` | `1e-10` | Minimum std threshold for constant-series rejection |
| `MAX_LATENCY_SAMPLES` | `1000` | Sliding window size for latency percentiles |

## Documentation

- [Docs Index](docs/README.md)
- [Roadmap](docs/project/ROADMAP.md)
- [Architecture](docs/project/ARCHITECTURE.md)
- [API Responses](docs/project/API_RESPONSES.md)
- [AI Usage](docs/ai/LLM_USAGE.md)
- [Git Protocol](docs/process/GIT_PROTOCOL.md)
