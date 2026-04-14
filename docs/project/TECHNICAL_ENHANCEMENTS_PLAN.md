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

## STANDALONE PROGRAM — Scalability and Multi-Detector Refactor

This scope moved to a dedicated implementation document:
`docs/project/SCALABILITY_REFACTOR_PLAN.md`

Reason:
- cross-cutting architectural impact (domain/services/repository/api/migration)
- better tracked as an independent rollout program than a single roadmap stage

---

## STAGE B (P1) — Industrial Sensor Validation Extensions

### Goal

Add real-world sensor quality rules (flat line and temporal gap) to strengthen data validation for industrial time series in the training path.

Rationale:
- directly addresses common IoT failure modes
- improves training-data quality and detector reliability
- shows practical production awareness beyond toy validation rules
- keeps inference path (`/predict`) lightweight and unaffected by training-only quality gates

### Implementation Tasks (Ordered)

1. Config extensions
   - Add in `app/config.py`:
     - `FLAT_LINE_WINDOW` (default 10)
     - `MAX_TEMPORAL_GAP_FACTOR` (default 2.0)
   - Add rule mode defaults in `app/config.py` (all default `off`):
     - `FLAT_LINE_MODE=off|warn|error`
     - `TEMPORAL_GAP_MODE=off|warn|error`

2. Domain exceptions
   - Add:
     - `FlatLineDetectedError`
     - `TemporalGapDetectedError`
   - Both subclass `ValidationServiceError`.

3. Validation service rules
   - Update `app/services/validation_service.py` to add:
     - rule: flat-line on trailing window
     - rule: max interval > factor x median interval
   - Keep fail-fast ordering deterministic.
   - Apply these rules only during `/fit` (training validation path), not `/predict`.
   - Define explicit config contract before implementation:
     - `RuleMode` enum: `off|warn|error`
     - `RuleConfig` typed contract (dataclass/TypedDict) with:
       - `flat_line_mode: RuleMode`
       - `temporal_gap_mode: RuleMode`
   - Support per-request optional rule-mode overrides (default `off`) by building `RuleConfig` in the route handler and passing it into service:
     - `ValidationService.validate_training_data(data, rule_config)`
   - Define a scalable rule-evaluation contract to support future rules without route rewrites:
     - shared enum-like mode: `off|warn|error`
     - shared evaluator return type for warnings:
       - `ValidationWarning` typed contract with `rule_id`, `mode`, `message`
     - centralized rule registry/list executed by validation service
     - each rule consults `RuleConfig` and either:
       - skips (`off`)
       - emits typed warning (`warn`)
       - raises typed exception (`error`)

4. Error handler mappings
   - Map new exceptions in `app/api/error_handlers.py` with codes:
     - `FLAT_LINE_DETECTED`
     - `TEMPORAL_GAP_DETECTED`
   - Errors are emitted only when the effective mode for the triggered rule is `error`.

5. API contract updates (training endpoint)
   - Extend `POST /fit/{series_id}` request with optional **query params** (default `off`), one per rule:
     - `flat_line_mode: off|warn|error`
     - `temporal_gap_mode: off|warn|error`
   - Keep backward compatibility when params are omitted.
   - Precedence rule (explicit):
     - request query override > environment/config default > hardcoded fallback
   - Invalid mode handling (explicit):
     - any mode value outside `off|warn|error` returns `422`
     - normalized error code: `INVALID_VALIDATION_MODE`
     - message must include accepted values
   - `warn` mode behavior:
     - request succeeds normally (no 4xx)
     - warnings are emitted in logs with required fields:
       - `event=training_validation_warning`
       - `rule_id`
       - `mode`
       - `series_id`
       - `request_id`
       - `message`
     - warnings are also returned in API response via additive optional field:
       - `warnings: list[ValidationWarning]` (optional)
     - required fields in base response contract remain unchanged
   - Canonical usage example:
     - `POST /fit/sensor_A?flat_line_mode=warn&temporal_gap_mode=error`

6. Tests
   - Add unit tests in `tests/unit/test_validation_service.py`:
     - one passing + one rejecting case per new rule
     - mode behavior per rule (`off`, `warn`, `error`)
   - Add integration tests for `/fit`:
     - default `off` does not reject due to new rules
     - `warn` does not reject request, logs warning, and returns `warnings` payload
     - `error` rejects with normalized code
   - Validate configurability through injected config values.
   - Validate per-request override precedence over config defaults.
   - Validate `/fit` query-parameter parsing for mode values and invalid-mode rejection.

7. Rule taxonomy for future extensibility
   - Define stable rule identifiers:
     - `flat_line`
     - `temporal_gap`
   - Define standard rule config keys:
     - `<RULE_ID>_mode`
     - rule-specific thresholds (e.g. `flat_line_window`, `max_temporal_gap_factor`)
   - Define stable typed warning/error payload attributes for future rules:
     - `rule_id`
     - `mode`
     - `message`
     - `error_code` (for `error` mode)
   - Treat this taxonomy as code-level contract (enum + typed class), not naming convention only.

8. Documentation updates (required)
   - Update `README.md` with mode-query examples for `/fit`.
   - Update `docs/project/API_RESPONSES.md` with:
     - `INVALID_VALIDATION_MODE`
     - `FLAT_LINE_DETECTED`
     - `TEMPORAL_GAP_DETECTED`
   - Update `docs/context/openapi_spec.yaml` as additive extension, or add explicit note if OpenAPI base contract is intentionally kept unchanged.

### Acceptance Criteria

- Scope is explicit: this stage affects only training (`/fit`) validation.
- New rule modes are optional request parameters on `/fit`, default `off`.
- Flat-line and temporal-gap rules support `off|warn|error` independently.
- In `error` mode:
  - flat-line trailing windows are rejected with `FLAT_LINE_DETECTED`
  - temporal gaps beyond configured factor are rejected with `TEMPORAL_GAP_DETECTED`
- In `warn` mode, requests succeed and warnings are logged.
- In `warn` mode, requests succeed, warnings are logged, and warnings are returned in optional response field.
- In `off` mode, rule is skipped.
- Thresholds and default modes are configurable through environment/config.
- Validation architecture supports adding future rules via shared mode contract/registry pattern.
- `RuleMode`, `RuleConfig`, and `ValidationWarning` are implemented as typed code contracts.
- `pytest -v` passes.

---

## STAGE C (P1) — Detector Comparison Benchmark Script

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

## STAGE D (P1) — Structured ML Logging for Operability

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

## STAGE E (P1) — Coverage Quality Gate and Developer UX

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

## STAGE F (P1) — Quality Tooling and Makefile Ergonomics

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
