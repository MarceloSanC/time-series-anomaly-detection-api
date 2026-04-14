# ENHANCEMENTS ROADMAP — POST v1.0.0

## Scope

This document tracks high-impact enhancements to improve operability, observability, and maintainability without overengineering.

Engineering rationale:
- Reduce operational blind spots by exposing inspectable system state.
- Speed up debugging and incident response with better runtime visibility.
- Improve API ergonomics for human operators and integrators.
- Keep extension points clean so future growth does not require architectural rewrites.

Constraints:
- preserve existing behavior of `/fit`, `/predict`, `/healthcheck`
- preserve current persistence/versioning model
- avoid breaking the base OpenAPI-defined contract for core endpoints

---

## Prioritization Model

- `P0`: must-have foundation for reliable operation
- `P1`: high-impact improvement with low-to-medium effort
- `P2`: nice-to-have improvements after core value is delivered

---

## STAGE A (P0) — Model Introspection Endpoints

### Goal

Expose model inventory and per-series metadata through API introspection:
- `GET /models`
- `GET /models/{series_id}`
- `GET /models/{series_id}/versions/{version}`

Rationale:
- removes filesystem dependency for state inspection
- improves operability and auditability during troubleshooting
- makes version lineage visible to users and integrators

### API Design (Additive Extension)

- `GET /models`
  - list of summaries:
    - `series_id`
    - `latest_version`
    - `n_samples`
    - `trained_at`

- `GET /models/{series_id}`
  - detail view:
    - `series_id`
    - `latest_version`
    - `versions`
    - `trained_at`
    - `n_samples`
    - `data_quality`:
      - `n_samples`
      - `mean`
      - `std`
      - `min_value`
      - `max_value`
      - `time_span_seconds`
      - `points_per_second`

- `GET /models/{series_id}/versions/{version}`
  - version metadata view (reproducibility-friendly):
    - `version`
    - `mean`
    - `std`
    - `n_samples`
    - `trained_at`
    - `training_duration_ms`
    - `data_range`
  - optional query:
    - `include_data=true` to include `training_data` when explicitly requested
  - default behavior:
    - do not return `training_data` (payload safety)

Data quality note:
- Keep `/fit` response unchanged to preserve core contract expectations.
- Expose data quality through introspection endpoints (`/models/{series_id}`) instead.

### Tasks (Ordered)

1. Response model alignment
   - Reuse `ModelInfo` where possible.
   - Add route-local response schemas only when mapping clarity requires it.
   - Files: `app/domain/schemas.py` (if needed), `app/api/routes/models.py`.

2. Route implementation
   - Create `app/api/routes/models.py`.
   - Implement:
     - `GET /models` using `ModelService.list_series()` and metadata mapping.
     - `GET /models/{series_id}` using `ModelService.get_series_info(series_id)`.
     - `GET /models/{series_id}/versions/{version}` using repository metadata lookup.
   - Add structured logs for list/detail requests.

3. Data quality mapping on detail endpoint
   - Derive and expose quality indicators from persisted metadata/training data:
     - `n_samples`, `mean`, `std`, `min_value`, `max_value`
     - `time_span_seconds = max_timestamp - min_timestamp`
     - `points_per_second = n_samples / max(time_span_seconds, 1)`
   - Keep calculation deterministic and side-effect free.

4. Version metadata payload design
   - Build default summary response without `training_data`.
   - Support `?include_data=true` for explicit full metadata retrieval.
   - Keep field names aligned with persisted metadata semantics.

5. Router wiring
   - Register models router in `app/api/routes/__init__.py`.

6. Error handling consistency
   - Unknown series in `/models/{series_id}` must return normalized `SERIES_NOT_FOUND`.
   - Unknown version in `/models/{series_id}/versions/{version}` must return normalized `VERSION_NOT_FOUND`.
   - Reuse existing global handlers.

7. Tests
   - Integration:
     - `/models` empty
     - `/models` with multiple series
     - `/models` tolerates incomplete metadata by default (`strict=false`) and still returns valid series
     - `/models?strict=true` returns `422 INCOMPLETE_MODEL_METADATA` when any latest metadata is missing/incomplete
     - `/models?strict=true` fail-fast behavior: if one series is incomplete and another is valid, response must be `422` (no partial list payload)
     - `/models/{series_id}` success
     - `/models/{series_id}` unknown -> `404 SERIES_NOT_FOUND`
     - `/models/{series_id}/versions/{version}` success summary
     - `/models/{series_id}/versions/{version}?include_data=true` includes `training_data`
     - `/models/{series_id}/versions/{version}?include_data=true` with legacy metadata missing `training_data` returns `training_data: []` (no 500)
     - `/models/{series_id}/versions/{version}` must exclude `training_data` by default
     - `/models/{series_id}/versions/{version}` unknown version -> `404 VERSION_NOT_FOUND`
     - `/models/{series_id}` includes `data_quality` with expected fields and sane values
     - `/models/{series_id}` validates `data_quality` calculations:
       - `time_span_seconds` is non-negative
       - `points_per_second` uses divisor `max(time_span_seconds, 1)`
       - `min_value <= mean <= max_value` for monotonic training fixture
     - `/models/{series_id}` with `min_timestamp == max_timestamp` yields `time_span_seconds == 0` and `points_per_second == n_samples/1`
   - Unit tests only for new helpers/mappers.
   - Suggested unit tests (new helper/mappers):
     - `_build_data_quality` with normal metadata payload
     - `_build_data_quality` with missing/empty `training_data` fallback behavior
     - `list_model_summaries(strict=True)` raises `MetadataIncompleteError` for missing metadata
     - `list_model_summaries(strict=False)` skips incomplete series and returns remaining summaries

8. Documentation
   - Add `/models*` usage examples in `README.md`.
   - Mark these endpoints as additive introspection extensions.

### Acceptance Criteria

- `curl --fail-with-body -sS http://localhost:8000/models` returns list payload.
- `curl --fail-with-body -sS http://localhost:8000/models/sensor_XYZ` returns detail payload.
- `curl --fail-with-body -sS http://localhost:8000/models/sensor_XYZ/versions/v1` returns version metadata summary.
- `/models/{series_id}` detail includes complete `data_quality` object.
- Unknown series returns normalized `SERIES_NOT_FOUND`.
- Unknown version returns normalized `VERSION_NOT_FOUND`.
- Core endpoint contracts remain unchanged.
- `pytest -v` passes.

---

## STAGE B (P0) — Multi-Detector Extensibility (Isolation Forest)

### Goal

Enable multiple anomaly detectors in the same architecture while preserving backward compatibility with the default gaussian detector.

Rationale:
- demonstrates true extensibility of the ML-serving architecture
- enables model-family comparison per `series_id` without redesign
- aligns with production MLE expectation of iterative detector evolution

### Implementation Tasks (Ordered)

1. Dependency and detector type definition
   - Add `scikit-learn` to `pyproject.toml`.
   - Define supported detector types (`gaussian`, `isolation_forest`) in a single source of truth.

2. Domain model implementation
   - Add `IsolationForestDetector` in `app/domain/models.py`.
   - Keep interface parity with `AnomalyDetectionModel` (`fit`, `predict`).

3. Service layer extension
   - Extend `ModelService.train(...)` and `ModelService.predict(...)` with `detector` parameter.
   - Preserve default `detector="gaussian"` when omitted.
   - Ensure version resolution is scoped per `(series_id, detector)`.

4. Repository storage evolution
   - Update `app/repository/model_repository.py` layout:
     - `storage/{series_id}/{detector}/{version}/model.joblib`
     - `storage/{series_id}/{detector}/{version}/metadata.json`
     - `storage/{series_id}/{detector}/index.json`
   - Provide compatibility path/migration guard where necessary.

5. API layer updates
   - Add optional `?detector=` to:
     - `POST /fit/{series_id}`
     - `POST /predict/{series_id}`
   - Validate unsupported detector values with normalized error response.

6. Tests
   - Unit tests for both detectors (`fit` and `predict` paths).
   - Integration tests for:
     - default gaussian behavior unchanged
     - isolation forest train/predict success
     - coexistence of both detectors under same `series_id`

### Acceptance Criteria

- `POST /fit/sensor_A?detector=isolation_forest` trains and persists.
- `POST /predict/sensor_A?detector=isolation_forest` returns prediction.
- `POST /fit/sensor_A` still uses gaussian by default.
- Both detector families coexist for same `series_id` without interference.
- `pytest -v` passes.

---

## STAGE C (P1) — Industrial Sensor Validation Extensions

### Goal

Add real-world sensor quality rules (flat line and temporal gap) to strengthen data validation for industrial time series in the training path.

Rationale:
- directly addresses common IoT failure modes (sensor disconnect, data loss)
- improves training-data quality and detector reliability
- demonstrates that the existing validation architecture absorbs new rules without structural changes
- keeps inference path (`/predict`) lightweight and unaffected by training-only quality gates

### Implementation Tasks (Ordered)

1. Config extensions
   - Add to `app/config.py` under the existing validation thresholds:
     - `flat_line_window: int` (default `10`)
     - `max_temporal_gap_factor: float` (default `2.0`)
   - No mode fields (`flat_line_mode`, `temporal_gap_mode`) — intentionally
     excluded from this scope.

2. Domain exceptions
   - Add to `app/domain/exceptions.py`:
     - `FlatLineDetectedError`
     - `TemporalGapDetectedError`
   - Both subclass `ValidationServiceError` — compatible with the existing
     handler dispatch in `app/api/error_handlers.py`, provided the error codes
     are registered in `VALIDATION_ERROR_CODE_MAP` (see Task 3).

3. Error handler mappings
   - Add to `VALIDATION_ERROR_CODE_MAP` in `app/api/error_handlers.py`:
     - `FlatLineDetectedError` → `"FLAT_LINE_DETECTED"`
     - `TemporalGapDetectedError` → `"TEMPORAL_GAP_DETECTED"`
   - No new handler functions required.

4. Validation service rules
   - Extend `ValidationService.__init__` with optional parameters:
     - `flat_line_window`
     - `max_temporal_gap_factor`
   - Defaults for both must come from `settings`, matching the current constructor pattern.
   - Append two rules to `app/services/validation_service.py`, after the
     existing five, in fail-fast order:
     - **Rule 6 — flat-line:** if `len(points) >= flat_line_window` and
       `max(values[-flat_line_window:]) == min(values[-flat_line_window:])`,
       raise `FlatLineDetectedError`.
     - **Rule 7 — temporal gap:** runs after timestamp uniqueness and ordering
       checks (rules 4–5) are guaranteed to have passed; compute pairwise
       intervals; if `max(intervals) > max_temporal_gap_factor × np.median(intervals)`,
       raise `TemporalGapDetectedError`.
   - Read thresholds from `self` attributes injected via `__init__`
     (same pattern as `min_data_points` and `std_threshold`).
   - Guard temporal-gap rule: skip if `len(points) < 2` (no intervals to compute).
   - `validate_training_data` signature and return type remain unchanged (`-> None`).
   - No changes to `ModelService`, `fit.py`, or `openapi_spec.yaml`.

5. Enable/disable flags
   - Add to `app/config.py`:
     - `flat_line_enabled: bool` (default `False`)
     - `temporal_gap_enabled: bool` (default `False`)
   - Extend `ValidationService.__init__` with:
     - `flat_line_enabled: bool | None = None`
     - `temporal_gap_enabled: bool | None = None`
   - Both default to `settings` values, matching the existing constructor pattern.
   - Add opt-in guard at the start of each rule in `validate_training_data`:
     - Rule 6 runs only if `self.flat_line_enabled` is `True`.
     - Rule 7 runs only if `self.temporal_gap_enabled` is `True`.
   - Update `.env.example` with the two new fields under the existing threshold block.

6. Tests
   - Add to `tests/unit/test_validation_service.py`:
     - flat-line disabled (`flat_line_enabled=False`): triggering series passes without rejection
     - flat-line enabled (`flat_line_enabled=True`): one passing case (trailing window not flat),
       one rejection case (trailing `flat_line_window` values identical, total series std above threshold)
     - temporal-gap disabled (`temporal_gap_enabled=False`): triggering series passes without rejection
     - temporal-gap enabled (`temporal_gap_enabled=True`): one passing case (uniform intervals),
       one rejection case (one gap > `max_temporal_gap_factor × np.median(intervals)`)
   - Inject all parameters via `ValidationService(...)` constructor — no config patching required.

7. Documentation updates
   - Update FastAPI route docs (`summary/description/responses`) with:
     - `FLAT_LINE_DETECTED`
     - `TEMPORAL_GAP_DETECTED`
   - Add note in `README.md` under the validation section: rules are opt-in via config,
     disabled by default, with configurable thresholds.

8. Add `ValidationService` extension points section to `docs/project/ARCHITECTURE.md`.

### Acceptance Criteria

- Scope is explicit: this stage touches only the training (`/fit`) validation path.
- `POST /fit` contract (request, response, status codes) is unchanged.
- Both rules are **disabled by default** (`flat_line_enabled=False`, `temporal_gap_enabled=False`).
- When enabled, series with `flat_line_window` or more identical trailing values is rejected
  with `400 FLAT_LINE_DETECTED`.
- When enabled, series with a gap exceeding `max_temporal_gap_factor × median interval` is
  rejected with `400 TEMPORAL_GAP_DETECTED`.
- All four parameters (`flat_line_window`, `max_temporal_gap_factor`, `flat_line_enabled`,
  `temporal_gap_enabled`) are configurable via `.env` / `config.py`.
- `ValidationService.__init__` accepts all four as optional overrides
  (consistent with existing constructor pattern).
- No changes to `ModelService`, `fit.py`, `openapi_spec.yaml`, or any other
  route or schema file.
- `pytest -v` passes.

---

## STAGE D (P1) — Detector Comparison Benchmark Script

### Goal

Provide a reproducible script to compare gaussian vs isolation forest detector behavior on the same dataset.

Rationale:
- demonstrates model validation discipline before production promotion
- converts architecture extensibility into measurable tradeoff analysis
- provides clear evidence for detector selection decisions

### Implementation Tasks (Ordered)

1. Script implementation
   - Create `scripts/compare_detectors.py`.
   - Build/generate realistic synthetic industrial dataset.
   - Train both detectors on same train split.
   - Evaluate on held-out split with injected anomalies.

2. Metrics and timing
   - Compute per detector:
     - true positive rate
     - false positive rate
     - inference latency (`p50`, `p95`, `p99`)
   - Add short conclusion field summarizing tradeoffs.

3. Artifact output
   - Save results to `scripts/detector_comparison.json`.
   - Print JSON summary to stdout for quick review.

4. Documentation integration
   - Reference results in:
     - `docs/project/MODELING_NOTES.md`
     - `README.md`

### Acceptance Criteria

- Script runs end-to-end without errors.
- Output JSON includes both detectors and all required metrics.
- Results are referenced in `MODELING_NOTES.md` and `README.md`.

### Validation Commands

```bash
.venv/bin/python scripts/compare_detectors.py
```

---

## STAGE E (P1) — Structured ML Logging for Operability

### Goal

Emit ML-context-aware structured logs that are directly usable in log aggregation systems.

Rationale:
- current code already sends rich `extra` fields, but default text formatter does not consistently expose them
- production troubleshooting needs searchable event-level metadata (`series_id`, `version`, `is_anomaly`, bounds, duration)
- enables reliable traceability across training/inference operations

### Implementation Tasks (Ordered)

1. Logging format toggle
   - Add configurable log format mode in config (e.g. `LOG_FORMAT=text|json`).
   - Keep `text` as default to preserve local developer readability.

2. JSON formatter support in logging utils
   - Extend `app/utils/logging.py` with a lightweight JSON formatter (no heavy dependency required).
   - Ensure common fields are always present:
     - `timestamp`, `level`, `logger`, `message`, `request_id`

3. ML event schema standardization
   - Normalize event names and payload keys in `app/services/model_service.py`:
     - `event=model_trained`
     - `event=prediction_served`
   - Include key ML fields:
     - train: `series_id`, `version`, `n_samples`, `mean`, `std`, `duration_ms`
     - predict: `series_id`, `version`, `value`, `mean`, `upper_bound`, `lower_bound`, `is_anomaly`

4. Compatibility and tests
   - Add unit tests for both logging modes:
     - text mode keeps current behavior
     - json mode emits parseable JSON with required keys
   - Keep API behavior unchanged.

5. Documentation
   - Add README section:
     - how to switch `LOG_FORMAT`
     - sample JSON log line
     - recommended fields for filtering in external log platforms
   - Add request-scoped summary logging checklist for:
     - `/fit`, `/predict`, `/plot`, `/models*`
   - Ensure prediction logs always include key decision fields.

### Acceptance Criteria

- With `LOG_FORMAT=json`, logs are valid JSON lines and include `request_id`.
- Training and prediction logs include standardized ML event fields.
- With default `LOG_FORMAT=text`, existing developer experience is preserved.
- `pytest -v` passes.

### Validation Commands

```bash
# default text logs
LOG_FORMAT=text .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

# json logs
LOG_FORMAT=json .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## STAGE F (P1) — Coverage Quality Gate and Developer UX

### Goal

Introduce a minimum automated coverage gate and an easy local coverage command.

Rationale:
- turns testability into an enforceable quality baseline
- prevents silent coverage regressions in future changes
- improves reviewer confidence with objective test-quality evidence

### Implementation Tasks (Ordered)

1. Pytest coverage gate in project config
   - Update `pyproject.toml` (`[tool.pytest.ini_options]`) to include:
     - `--cov=app`
     - `--cov-report=term-missing`
     - `--cov-fail-under=80`

2. Makefile coverage target
   - Add `coverage` target:
     - `pytest --cov=app --cov-report=html --cov-report=term-missing`
     - print report path (`htmlcov/index.html`)

3. Documentation
   - Add short README note:
     - coverage gate threshold (`>=80%`)
     - command to generate HTML report (`make coverage`)

### Acceptance Criteria

- `make coverage` runs successfully.
- Project fails CI/local test run if coverage drops below `80%`.
- README includes coverage gate and usage command.

### Validation Commands

```bash
make coverage
pytest -v
```

---

## STAGE G (P1) — Quality Tooling and Makefile Ergonomics

### Goal

Improve developer and evaluator experience with a modern linting toolchain and complete Make targets for common workflows.

Rationale:
- reduces evaluation friction (single-command operations)
- raises code-quality signal with ecosystem-standard tooling
- makes local/CI behavior more predictable and repeatable

### Implementation Tasks (Ordered)

1. Add Ruff as dev dependency
   - Update `pyproject.toml` optional dev dependencies with `ruff`.
   - Add minimal Ruff config under `[tool.ruff]` / `[tool.ruff.lint]`.

2. Makefile target hardening
   - Keep existing stable targets (`install`, `test`, `docker-up`, `docker-test`).
   - Add/standardize:
     - `docker-down` (`docker compose down -v`)
     - `benchmark` (`.venv/bin/python scripts/benchmark.py`)
     - `smoke` (`./scripts/manual/stage2_smoke_test.sh`)
     - `check` (aggregates lint + test + coverage where applicable)

3. Lint integration
   - Update `lint` target to use Ruff (`ruff check app tests`).
   - Optionally add `format-check` target using Ruff formatter in check mode.

4. Documentation
   - Add "Common Make Targets" section in `README.md` with brief usage examples.
   - Add short "Troubleshooting" section in `README.md`:
     - missing metadata for plot
     - known validation errors
     - quick Docker test commands

### Acceptance Criteria

- `make lint` runs with Ruff and exits cleanly.
- `make docker-down`, `make benchmark`, and `make smoke` run without command errors.
- `make check` provides a reliable pre-PR gate.
- README documents the new targets.

### Validation Commands

```bash
make lint
make test
make coverage
make docker-test
make benchmark
make smoke
```

---

## Delivery Notes

- Keep each stage in its own small PR.
- Prefer reusing existing service logic over adding repository complexity.
- Favor explicit mappings and readable code over abstraction depth.
