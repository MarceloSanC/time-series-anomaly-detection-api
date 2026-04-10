# ROADMAP — 5-DAY EXECUTION PLAN

## RULE ZERO

**No enhancement work begins before the MVP checklist in `skills.md` is 100% complete.**
This is non-negotiable. Do not rationalize exceptions.

---

## DAY 1 — Foundation and Core Domain

**Goal:** ModelService working end-to-end with tests. No API yet.

### Tasks (in order)

1. Project scaffolding
   - `pyproject.toml` with all dependencies pinned
   - `Makefile` with targets: `install`, `test`, `run`, `lint`, `docker-build`, `docker-up`
   - `.env.example` with all configurable parameters
   - `app/config.py` reading from `.env` via `pydantic-settings`

2. Domain layer
   - `app/domain/schemas.py` — all Pydantic models as defined in `skills.md`
   - `app/domain/models.py` — `AnomalyDetectionModel` with the iteration bug fixed silently

3. Repository layer
   - `app/repository/model_repository.py`
   - Methods: `save(series_id, version, model, metadata)`, `load(series_id, version)`, `get_index(series_id)`, `list_all()`, `version_exists(series_id, version)`
   - Atomic index write: write to `.tmp` then `os.replace`

4. Utils layer
   - `app/utils/concurrency.py` — `LockManager` with `threading.Lock` per `series_id`
   - `app/utils/logging.py` — structured logging with `request_id` context var

5. Service layer
   - `app/services/model_service.py` — `train(series_id, data)`, `predict(series_id, data_point, version=None)`, `list_series()`, `get_series_info(series_id)`
   - `app/services/metrics_service.py` — `record()`, `snapshot()`

6. Unit tests (Day 1 does not end without these passing)
   - `tests/unit/test_model_service.py` — train, predict, version increment, series not found
   - `tests/unit/test_model_repository.py` — save/load roundtrip, atomic write, version listing

### End of Day 1 Checkpoint

```bash
pytest tests/unit/ -v
# All tests must pass before Day 2 starts
```

---

## DAY 2 — API Layer and Docker

**Goal:** Full API running in Docker with all mandatory endpoints.

### Tasks (in order)

1. FastAPI application
   - `app/main.py` — app factory, lifespan (initialize `LockManager`, `MetricsService`), register routers, register error handlers, add request timing middleware, add `request_id` middleware

2. Dependency injection
   - `app/dependencies.py` — `get_model_service()`, `get_metrics_service()`, `get_lock_manager()`

3. Route handlers (one file per domain)
   - `app/api/routes/train.py` — `POST /train/{series_id}`
   - `app/api/routes/predict.py` — `POST /predict/{series_id}` + `?version=` query param
   - `app/api/routes/models.py` — `GET /models`, `GET /models/{series_id}`
   - `app/api/routes/health.py` — `GET /health`
   - `app/api/routes/metrics.py` — `GET /metrics`

4. Error handlers
   - `app/api/error_handlers.py` — catch domain exceptions, return `ErrorResponse` with correct status codes

5. Docker
   - `Dockerfile` — multi-stage build, non-root user
   - `docker-compose.yml` — mounts `./storage` as volume

6. Integration tests
   - `tests/integration/test_train_endpoint.py`
   - `tests/integration/test_predict_endpoint.py`
   - `tests/integration/test_metrics_endpoint.py`

7. Manual smoke test
   - Train 3 different `series_id` values
   - Predict on each
   - Verify versions increment on retrain
   - Verify `GET /models` lists all 3

### End of Day 2 Checkpoint

```bash
docker-compose up -d
curl http://localhost:8000/health
# Must return 200

pytest tests/ -v
# All tests must pass
```

---

## DAY 3 — Validation, Robustness, and Versioning Extensions

**Goal:** Preflight validation complete, versioning fully operational, error handling production-quality.

### Tasks (in order)

1. Validation service
   - `app/services/validation_service.py` — implement all 5 rules from `skills.md`
   - Wire into `model_service.train()` — validation runs before lock acquisition

2. Domain exceptions
   - Create `app/domain/exceptions.py` with typed exceptions: `InsufficientDataError`, `ConstantSeriesError`, `DuplicateTimestampsError`, `UnorderedTimestampsError`, `InvalidValuesError`, `SeriesNotFoundError`, `VersionNotFoundError`
   - Update `error_handlers.py` to catch each and return correct status + error code

3. Versioning extensions
   - Ensure `GET /predict/{series_id}?version=v2` works correctly
   - Ensure `GET /models/{series_id}` returns full version history with metadata per version

4. Unit tests for validation
   - `tests/unit/test_validation_service.py` — one passing + one rejection case per rule (10 tests minimum)

5. Review and harden
   - Verify all error responses match `ErrorResponse` schema
   - Verify no endpoint returns a raw Python exception under any input
   - Add `series_id` sanitization (reject path traversal characters: `/`, `..`, `\`)

### End of Day 3 Checkpoint

```bash
# Test constant series rejection
curl -X POST http://localhost:8000/train/sensor_flat \
  -H "Content-Type: application/json" \
  -d '{"data": [{"timestamp": 1, "value": 5.0}, ...]}'
# Must return 400 with error code CONSTANT_SERIES

pytest tests/ -v
# All tests pass
```

---

## DAY 4 — Performance Testing and Visualization

**Goal:** Benchmark results documented, plot endpoint working.

### Tasks (in order)

1. Benchmark script
   - `scripts/benchmark.py` — 100 parallel inference requests with `asyncio` + `httpx.AsyncClient`
   - Output: p50, p95, p99 latency, total throughput
   - Save to `scripts/benchmark_results.json`
   - Run against a pre-trained series with realistic data (>100 points)

2. Visualization endpoint
   - `app/api/routes/plot.py` — `GET /plot?series_id=X&version=v3`
   - `matplotlib.use('Agg')` — required for Docker headless
   - Load metadata from `metadata.json` — do NOT load `model.joblib` just for plotting
   - Plot: scatter of training points, mean line, upper/lower 3-sigma bounds
   - Return `StreamingResponse` with `media_type="image/png"`

3. Sample request scripts
   - `scripts/examples/train_request.sh` — curl command with realistic sample data
   - `scripts/examples/predict_request.sh` — curl command for single point prediction

4. README draft
   - Setup instructions (clone → docker-compose up)
   - All endpoint examples with curl
   - Benchmark results table
   - Known limitations section

### End of Day 4 Checkpoint

```bash
python scripts/benchmark.py
# Must complete without errors, output p95/p99

curl "http://localhost:8000/plot?series_id=sensor_XYZ" --output plot.png
# Must produce a valid PNG file
```

---

## DAY 5 — Polish, Documentation, and Delivery

**Goal:** Repository is clean, documented, and ready for evaluation.

### Tasks (in order)

1. Code review pass
   - Remove all dead code, commented-out blocks, debug prints
   - Ensure every public method has a type hint on all parameters and return
   - Ensure every public method has a brief docstring
   - Ensure no hardcoded values outside `config.py`

2. Test coverage check
   - Run `pytest --cov=app --cov-report=term-missing`
   - Identify and fill any gaps in critical paths

3. README finalization
   - Setup instructions verified from scratch (clone → up → test)
   - Usage examples for every endpoint
   - Benchmark results
   - Failure modes / known limitations
   - Architecture decisions (brief)
   - `.env.example` explanation

4. Docker final verification
   - `docker-compose down -v && docker-compose up --build`
   - Run full test suite against the container
   - Verify model persistence: stop container, start again, predict must still work

5. Repository cleanup
   - `.gitignore` covers: `__pycache__`, `.env`, `storage/*` (not `.gitkeep`), `*.pyc`, `.pytest_cache`
   - Commit history is clean and readable
   - Create release tag: `v1.0.0`

### Final Delivery Checklist

- [ ] `docker-compose up` works from zero
- [ ] `GET /health` returns 200
- [ ] Train + predict cycle works for 3 distinct `series_id`
- [ ] Retrain creates new version, old version still accessible
- [ ] `?version=v1` on a v3 series returns correct result
- [ ] Constant series rejected with 400
- [ ] Insufficient data rejected with 400
- [ ] `GET /metrics` returns latency data
- [ ] `/plot` returns PNG
- [ ] Benchmark ran and results in README
- [ ] All tests pass: `pytest -v`
- [ ] README has setup instructions that work
- [ ] `.env.example` present and complete
- [ ] No raw exceptions exposed to clients
- [ ] `request_id` in log lines

---

## TIME BUDGET (approximate)

| Day | Focus                        | Risk if skipped         |
|-----|------------------------------|-------------------------|
| 1   | Domain + Service + Tests     | Everything else breaks  |
| 2   | API + Docker + Integration   | Nothing is deliverable  |
| 3   | Validation + Error handling  | Fails evaluation rubric |
| 4   | Benchmark + Plot             | Missing optional points |
| 5   | Polish + Docs + Delivery     | Poor first impression   |

---

## DEPENDENCY STACK (pin these versions in pyproject.toml)

```toml
[project]
requires-python = ">=3.11"

dependencies = [
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.30.0",
    "pydantic>=2.7.0",
    "pydantic-settings>=2.3.0",
    "numpy>=1.26.0",
    "joblib>=1.4.0",
    "matplotlib>=3.9.0",
    "httpx>=0.27.0",       # for benchmark script and test client
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
]
```
