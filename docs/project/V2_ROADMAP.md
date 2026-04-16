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

Status: Completed

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

### Validation Evidence

- `/models`, `/models/{series_id}` e `/models/{series_id}/versions/{version}` implementados e cobertos por testes de integracao.
- Casos de erro normalizados validados (`SERIES_NOT_FOUND`, `VERSION_NOT_FOUND`, `INCOMPLETE_MODEL_METADATA` em `strict=true`).

---

## STAGE B (P0) — Multi-Detector Extensibility (Isolation Forest)

Status: Completed

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

### Validation Evidence

- Implemented via [MULTI_DETECTOR_PLAN](MULTI_DETECTOR_PLAN.md) (5 phases: domain → repository → service → API → docs).
- `?detector=` supported on `/fit`, `/predict`, and all `/models*` endpoints (exceeds Stage B scope).
- Detector-scoped storage layout: `storage/{series_id}/{detector}/{version}/`.
- Normalized errors: `422 UNSUPPORTED_DETECTOR`, `404 VERSION_NOT_FOUND_FOR_DETECTOR`.
- Test suite: unit + integration coverage for both detectors, coexistence, and error cases.
- `pytest -v` passing with 99%+ coverage.

---

## STAGE C (P1) — Industrial Sensor Validation Extensions

Status: Completed

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

### Validation Evidence

- Rules implemented in `app/services/validation_service.py` (Rules 6–7), disabled by default.
- Config flags: `FLAT_LINE_ENABLED=false`, `TEMPORAL_GAP_ENABLED=false` in `.env.example`.
- Error codes registered in `VALIDATION_ERROR_CODE_MAP`: `FLAT_LINE_DETECTED`, `TEMPORAL_GAP_DETECTED`.
- Unit tests injected via `ValidationService(...)` constructor — no config patching.
- `pytest -v` passing with 99%+ coverage.

---

## STAGE D (P1) — Detector Comparison Benchmark Script

Status: Completed

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
   - Note: IsolationForest is sensitive to low-variance training data (masking effect) — the synthetic dataset must have sufficient variance for a fair comparison.

3. Artifact output
   - Save results to `scripts/detector_comparison.json`.
   - Print JSON summary to stdout for quick review.

4. Documentation integration
   - Reference results in:
     - `docs/project/MODEL_DESIGN_NOTES.md`
     - `README.md`

### Acceptance Criteria

- Script runs end-to-end without errors.
- Output JSON includes both detectors and all required metrics.
- Results are referenced in `MODEL_DESIGN_NOTES.md` and `README.md`.

### Validation Commands

```bash
.venv/bin/python scripts/compare_detectors.py
```

### Conclusions

- Gaussian is expected to outperform IsolationForest on gaussian-distributed datasets.
  The Gaussian detector is a parametric model calibrated to the exact distribution of the
  synthetic data (mean/std threshold), while IsolationForest is a general-purpose density
  estimator that does not exploit distributional structure. Additionally, IsolationForest's
  threshold is set at the 10th percentile of training scores, which flags ~10% of normal
  points as anomalies (higher FPR), whereas the Gaussian 3-sigma bound flags ~0.13%.
  IsolationForest holds the advantage in non-gaussian, multimodal, or clustered-anomaly
  scenarios, and is the only option for detecting negative outliers (below the mean), which
  the Gaussian detector ignores by design.

---

## STAGE E (P1) — Structured ML Logging for Operability

Status: Completed

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
     - predict: `series_id`, `version`, `value`, `is_anomaly`, detector-specific decision fields
       (e.g. gaussian: `mean`, `upper_bound`; isolation_forest: `score_threshold`)

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

Status: Completed

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

### Validation Evidence

- `make coverage`: `Required test coverage of 80% reached. Total coverage: 99.73%`
- Test summary: `81 passed, 15 warnings in 5.64s`

---

## STAGE G (P1) — Quality Tooling and Makefile Ergonomics

Status: Completed

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
   - Keep lint scope minimal in this stage: default Ruff rule families `E` and `F` only.

2. Makefile target hardening
   - Keep existing stable targets (`install`, `test`, `docker-up`, `docker-test`).
   - Add/standardize:
     - `docker-down` (`docker compose down -v`)
     - `benchmark` (`.venv/bin/python scripts/benchmark.py`)
     - `smoke` (`./scripts/manual/stage2_smoke_test.sh`)
     - `check` with exact command chain: `make lint && make test`

3. Lint integration
   - Update `lint` target to use Ruff (`ruff check app tests`).
   - Keep `format-check` out of scope for this stage (evaluate later in a separate follow-up).

4. Documentation
   - Add "Common Make Targets" section in `README.md` with brief usage examples.
   - Add short "Troubleshooting" section in `README.md`:
     - missing metadata for plot
     - known validation errors
     - quick Docker test commands

### Acceptance Criteria

- `make lint` runs with Ruff and exits cleanly.
- `make docker-down`, `make benchmark`, and `make smoke` run without command errors.
- `make check` runs `make lint && make test` and provides a reliable pre-PR gate.
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

## STAGE H (P1) — Industrial Dataset Generator

Status: Pending

### Goal

Produce a realistic synthetic industrial dataset generator and wire it into the existing analysis scripts.

Rationale:
- current analysis scripts (compare_detectors, drift_analysis) use simple gaussian noise; a realistic dataset with degradation trends, regime shifts, and electrical spikes validates detector behavior under conditions closer to production IoT data
- the generator transforms theoretical claims in MODEL_DESIGN_NOTES.md into empirically grounded evidence backed by domain-representative data

### Implementation Tasks (Ordered)

1. Industrial dataset generator
   - Create `scripts/generate_industrial_dataset.py`.
   - Implement `generate_vibration_series()` with configurable components:
     - base signal with slow degradation trend (`base + degradation_rate × t`)
     - seasonality (`amplitude × sin(2π × t / period)`)
     - gaussian noise
     - sudden regime changes (mean shift at random points)
     - electrical spikes (single points at ±10σ from local mean)
     - flat line segments (simulate sensor disconnect)
     - injected ground-truth anomalies (sustained deviations of +4σ for 3–5 consecutive points)
   - Save output to `data/industrial_sample.json` with schema:
     ```json
     {
       "metadata": { "n_points": int, "sampling_hz": float, "n_injected_anomalies": int },
       "series": [ { "timestamp": int, "value": float, "is_anomaly": bool } ]
     }
     ```
   - Document parameters and usage in script docstring.

2. Wire generator into existing analysis scripts
   - Update `scripts/compare_detectors.py` to optionally load from `data/industrial_sample.json` instead of generating inline gaussian data.
   - Update `scripts/drift_analysis.py` to reference generator for the baseline and drift datasets.
   - Keep existing synthetic fallback when file is absent (backward compatible).

3. Tests
   - Unit test for `generate_vibration_series()`: verify output schema, anomaly label count, and presence of all configured components (regime changes, spikes, flat segments).

4. Documentation
   - Add "Sample Data" section in `README.md` referencing `generate_industrial_dataset.py` and `data/industrial_sample.json`.
   - Update MODEL_DESIGN_NOTES.md §4 to reference results from the generator-backed runs of compare_detectors and drift_analysis.

### Acceptance Criteria

- `python scripts/generate_industrial_dataset.py` produces `data/industrial_sample.json` with all required schema fields.
- `compare_detectors.py` and `drift_analysis.py` consume the generated file when present.
- `pytest -v` passes.

---

## STAGE I (P1) — Enriched `/plot` Endpoint with Detector-Aware Rendering

Status: Pending

### Goal

Extend the `/plot` endpoint to support both detectors with separate, intrinsics-aware rendering functions, and enrich the visual output with anomaly coloring and trend overlay.

Rationale:
- `/plot` currently resolves only to the gaussian namespace and renders mean/std bands, which are undefined for isolation_forest; this makes the endpoint silently incorrect for isolation_forest series
- separating rendering per detector allows each plot to expose the model's own decision surface — the threshold bands for gaussian, the score boundary for isolation_forest — making the diagnostic value of the plot proportional to the model's actual behavior
- anomaly coloring and trend line convert a static scatter into an actionable view of which training points the model flagged and how the signal evolves over time

### Implementation Tasks (Ordered)

1. Add `?detector=` param to `/plot` route
   - Add optional `detector: str = "gaussian"` query parameter to the route handler in `app/api/routes/plot.py`.
   - Keep detector validation centralized in `ModelService` (same pattern used by `/fit`, `/predict`, `/models*`) so unrecognized values return normalized `UNSUPPORTED_DETECTOR` (422).
   - Propagate detector through `model_service.get_plot_data(series_id, version, detector)` and into `_resolve_version`.
   - When `version` is explicitly provided, version lookup must remain detector-scoped:
     - if the version does not exist in the selected detector namespace, return `404 VERSION_NOT_FOUND_FOR_DETECTOR`.

2. Separate rendering functions per detector
   - Refactor `app/api/routes/plot.py` to extract two rendering functions:
     - `_render_gaussian_plot(fig, ax, plot_data)` — retain current behavior: scatter of training points, `mean` horizontal line, `upper_bound` (mean + 3σ) and `lower_bound` (mean − 3σ) band lines.
     - `_render_isolation_forest_plot(fig, ax, plot_data)` — isolation-forest-specific elements:
       - scatter training points colored by anomaly score intensity (blue = low score, red = high score) using `training_scores: list[float]` from metadata; fall back to uniform color when absent.
       - `score_threshold` horizontal line (the decision boundary, equivalent to upper_bound for gaussian).
       - subtitle/title annotation with contamination value:
         - numeric contamination: format as percent (`{contamination:.1%}`)
         - `"auto"` contamination: render as literal (`contamination=auto`)
   - Route handler dispatches to the correct function based on `detector`.

3. Persist `training_scores` for isolation_forest
   - After fitting an isolation_forest model, compute the anomaly score for each training point using `score_samples` (same score space used by `score_threshold`).
   - Store as `training_scores: list[float]` in `metadata.json` alongside `training_data`.
   - No change needed for gaussian (gaussian does not use scores).

4. Anomaly coloring for both detectors
   - Extend the model_service training path: after fitting either detector (gaussian and isolation_forest), run the same post-fit pass over training data to compute `training_anomaly_flags: list[bool]` using `model.predict(...)` on each training point.
   - Persist `training_anomaly_flags` in `metadata.json` alongside `training_data`.
   - In `_render_gaussian_plot`: use `training_anomaly_flags` for blue/red point coloring when present; fall back to uniform color (backward compatible).
   - In `_render_isolation_forest_plot`: use `training_scores` for color intensity when present; overlay `training_anomaly_flags` as marker shape (circle = normal, x = flagged) when available.

5. Trend line and title annotation (both detectors)
   - Add linear regression trend line over `(timestamp, value)` pairs as a thin dashed overlay in both rendering functions.
   - Add point count annotation to the plot title: `Series {series_id} ({version}) — {n} points`.
   - Use only `numpy` (already a dependency) for the regression.

6. Tests
   Integration (`tests/integration/test_plot_endpoint.py`):
   - `/plot?detector=gaussian` after normal `/fit`: verify `200`, `image/png`, PNG signature (exercises blue/red coloring path).
   - `/plot?detector=isolation_forest` after `/fit?detector=isolation_forest`: verify `200`, `image/png`, PNG signature (exercises score-colored scatter + flags overlay).
   - `/plot` without detector param: verify gaussian default behavior unchanged (`200`, `image/png`, PNG signature).
   - `/plot?detector=isolation_forest&version=v1` where v1 exists only in gaussian namespace: verify `404 VERSION_NOT_FOUND_FOR_DETECTOR`.
   - `/plot?detector=isolation_forest` where series was only trained with gaussian (no IF namespace): verify `404 SERIES_NOT_FOUND`.
   - `/plot?detector=random_forest`: verify `422 UNSUPPORTED_DETECTOR`.
   - Backward compat — gaussian legacy metadata without `training_anomaly_flags`: manually save metadata without the field, verify `200` and PNG signature (uniform color fallback, no crash).
   - Backward compat — isolation_forest legacy metadata without `training_scores`/`training_anomaly_flags`: manually save IF metadata without those fields, verify `200` and PNG signature (uniform color fallback, no score overlay).

   Unit/service (`tests/unit/test_model_service.py`):
   - `get_plot_data(detector="gaussian")`: verify payload contains `training_anomaly_flags` as a list of bools with length equal to `n_samples`, and `mean`/`std` are present.
   - `get_plot_data(detector="isolation_forest")`: verify payload contains `score_threshold` (float), `training_scores` (list of floats, length equal to `n_samples`), `training_anomaly_flags` (list of bools), and `contamination`.
   - `get_plot_data(detector="isolation_forest")` for series trained only with gaussian: verify `SeriesNotFoundError` is raised.

7. Documentation
   - Update `/plot` section in `ARCHITECTURE.md` to document the detector param and per-detector rendering behavior.
   - Add a brief `/plot?detector=` example in `README.md` endpoint examples section.

### Acceptance Criteria

- `GET /plot?series_id={id}` (no detector param) behaves identically to before this stage.
- `GET /plot?series_id={id}&detector=isolation_forest` returns `200` and renders score-colored points with a `score_threshold` line.
- `GET /plot?series_id={id}&detector=gaussian` renders mean, upper_bound, lower_bound lines with anomaly-colored points when `training_anomaly_flags` present.
- `GET /plot?series_id={id}&detector={d}&version={v}` resolves version strictly inside selected detector namespace; mismatches return `404 VERSION_NOT_FOUND_FOR_DETECTOR`.
- Unsupported detector param returns `422` with `UNSUPPORTED_DETECTOR`.
- `pytest -v` passes.

---

## STAGE J (P2) — Production Observability Stack

Status: Pending

### Goal

Add opt-in production observability tooling: a Prometheus + Grafana monitoring stack and a Kafka streaming inference example.

Rationale:
- the MLE Production Systems role explicitly evaluates familiarity with monitoring infrastructure and event-driven platforms
- both items are additive only and do not modify the core application; they are architectural demonstrations, not production features
- Prometheus/Grafana converts the in-memory latency data that already exists into a persistent, queryable time-series store; Kafka shows that the service layer is correctly decoupled from HTTP transport

### Implementation Tasks (Ordered)

1. Prometheus metrics endpoint
   - Create `app/api/routes/metrics_prometheus.py` with `GET /metrics`.
   - Format output manually from `MetricsService.snapshot()` in Prometheus text format (no `prometheus_client` dependency).
   - Expose: request count per endpoint, p50/p95/p99 latency per endpoint, anomaly detection rate per series.
   - Register route in `app/main.py` only when `PROMETHEUS_ENABLED=true` (env flag, default `false`).

2. Prometheus + Grafana compose stack
   - Create `docker-compose.observability.yml` (separate file — do NOT modify `docker-compose.yml`).
   - Services: Prometheus (scrapes `/metrics`) + Grafana (pre-provisioned dashboard).
   - Create `monitoring/prometheus.yml` with scrape config pointing to the API service.
   - Create `monitoring/grafana/dashboard.json` with panels:
     - request rate per endpoint
     - p95 inference latency over time
     - anomaly detection rate per series
     - training events timeline
   - Startup command: `docker compose -f docker-compose.yml -f docker-compose.observability.yml up`.

3. Kafka streaming inference example
   - Create `examples/kafka/schemas.py` — message schemas mirroring existing `DataPoint` and `PredictionResponse` Pydantic models.
   - Create `examples/kafka/producer.py` — generates synthetic sensor data and publishes to input topic `raw_sensor_data`.
   - Create `examples/kafka/consumer.py` — reads from `raw_sensor_data`, calls `ModelService.predict()` directly (not via HTTP), publishes result to `anomaly_events` topic.
   - Create `docker-compose.kafka.yml` with Kafka + Zookeeper + consumer + producer services.
   - Startup command: `docker compose -f docker-compose.yml -f docker-compose.kafka.yml up`.
   - The consumer must import `ModelService` directly — this validates that the service layer is transport-agnostic.

4. Documentation
   - Add "Monitoring" section in `README.md` documenting the observability stack and startup command.
   - Add "Advanced: Streaming" section in `README.md` documenting the Kafka example, clearly scoped as an architectural example.
   - Both sections must explicitly state these are opt-in and do not affect the core application.

### Acceptance Criteria

- `docker compose -f docker-compose.yml -f docker-compose.observability.yml up` starts without error.
- Grafana dashboard at `localhost:3000` shows live data after running `make benchmark`.
- `docker compose -f docker-compose.yml -f docker-compose.kafka.yml up` starts without error.
- Running `producer.py` generates messages visible in the `anomaly_events` topic via `consumer.py`.
- Main `docker-compose.yml` is completely unchanged.
- Core application test suite (`pytest -v`) passes without any observability dependencies installed.

---

## Delivery Notes

- Keep each stage in its own small PR.
- Prefer reusing existing service logic over adding repository complexity.
- Favor explicit mappings and readable code over abstraction depth.
