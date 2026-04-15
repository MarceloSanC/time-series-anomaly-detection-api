# MULTI-DETECTOR REFACTOR PLAN

## Purpose

This document defines the architectural foundation required before implementing a second detector.
It isolates the cross-cutting refactor from the feature work (IsolationForest, Stage B in V2_ROADMAP).

Why separate:
- scope is cross-cutting (domain, services, repository, API, docs, tests, migration)
- changes affect persistence contracts and runtime resolution semantics
- Stage B cannot be implemented safely without this foundation in place

---

## Target Outcomes

1. Detector-oriented architecture with explicit contracts.
2. Deterministic runtime resolution for `(series_id, detector, version)`.
3. Backward-compatible default behavior (`gaussian`) on core endpoints.
4. Safe migration from legacy storage layout.
5. Full regression coverage for pre- and post-migration flows.

---

## Out of Scope

- distributed queues/orchestrators
- external shared model store (S3, DB, Redis, etc.)
- horizontal scaling redesign of deployment topology

---

## Architectural Decisions (Frozen)

1. Detector namespace is explicit and required for non-default behavior.
2. Default detector remains `gaussian`.
3. Version resolution is detector-scoped only (no cross-detector fallback).
4. Storage/index is detector-scoped.
5. No migration required — storage is empty at implementation time. The detector-scoped layout `{series_id}/{detector}/{version}/` is adopted unconditionally from the start.

---

## Strategic Decisions

| Decision | Resolution |
|---|---|
| `DetectorType` representation | `Literal["gaussian", "isolation_forest"]` type alias in `schemas.py` — simple, Pydantic-native, no enum overhead at this scale |
| Detector interface contract | `Protocol` with `fit(data: TimeSeries)` and `predict(data_point: DataPoint) -> bool` — consistent with existing duck typing, no ABC overhead |
| Metadata schema | Common fields: `version`, `detector`, `n_samples`, `trained_at`, `training_duration_ms`, `data_range`. Detector-specific fields stored under a `model_params` key (gaussian: `mean`, `std`; isolation_forest: `n_estimators`, `contamination`, `score_threshold`) |
| `/models*` behavior with multiple detectors | Backward-compatible: no-detector view returns gaussian data by default; optional `?detector=` param to query other families |
| `FitResponse` / `PredictResponse` | Add `detector: str` field to both responses — required for comparison and traceability |
| Stage A `data_quality` for non-gaussian detectors | Fields `mean`, `std` become `Optional` — returned for gaussian, `null` for detectors that don't compute them |
| Migration strategy | No migration — storage is empty at implementation time. Detector-scoped layout adopted unconditionally from v1. Any migration references in scripts and docs are forward-compatibility notes only, not runtime behavior. `schema_version: "1"` written to `index.json` in Phase 2 as a forward-compatibility marker. |
| `IsolationForestDetector` threshold | `np.percentile(train_scores, 10)` — bottom 10% of training scores defines the anomaly cutoff. More robust than mean (resistant to extreme training outliers). Anomaly when `score_samples(value) < threshold`. |
| `IsolationForestDetector` params | `n_estimators=100`, `contamination="auto"`, `random_state=42` — not configurable at this stage. `score_threshold` persisted in `model_params` for inspection. |

---

## Module Impact Matrix

### Domain Layer

- `app/domain/models.py`
  - add `DetectorProtocol` with `fit` and `predict` interface
  - add `IsolationForestDetector` implementing `DetectorProtocol`
- `app/domain/schemas.py`
  - add `DetectorType = Literal["gaussian", "isolation_forest"]`
  - add `model_params` field to metadata schema (detector-specific)
  - make `mean`, `std` optional in `data_quality` response
  - add `detector` field to `FitResponse` and `PredictResponse`
- `app/domain/exceptions.py`
  - add `UnsupportedDetectorError`
  - add `VersionNotFoundForDetectorError`

### Repository Layer

- `app/repository/model_repository.py`
  - detector-aware paths: `storage/{series_id}/{detector}/{version}/`
  - detector-aware save/load/index/version methods

### Service Layer

- `app/services/model_service.py`
  - extend `train` and `predict` with `detector: DetectorType = "gaussian"`
  - resolve version inside selected `(series_id, detector)` namespace
  - keep all existing list/info flows backward-compatible

### API Layer

- `app/api/routes/fit.py` — optional `?detector=` query param, default `gaussian`
- `app/api/routes/predict.py` — optional `?detector=` query param, default `gaussian`
- `app/api/routes/models.py` — optional `?detector=` query param, default `gaussian`
- `app/api/error_handlers.py`
  - `UnsupportedDetectorError` → `UNSUPPORTED_DETECTOR` (422)
  - `VersionNotFoundForDetectorError` → `VERSION_NOT_FOUND_FOR_DETECTOR` (404)

### Configuration and Dependencies

- `pyproject.toml` — add `scikit-learn`
- `app/config.py` — no new fields required at this stage

### Docs and Scripts

- `docs/project/ARCHITECTURE.md` — detector contract, storage contract, migration behavior
- `README.md` — detector query examples and compatibility notes
- `scripts/compare_detectors.py` — benchmark script (Stage D, after Stage B lands)

---

## Storage Contract (Target)

```text
storage/
  {series_id}/
    gaussian/
      index.json
      v1/model.joblib
      v1/metadata.json
    isolation_forest/
      index.json
      v1/model.joblib
      v1/metadata.json
```

Detector index schema:

```json
{
  "series_id": "sensor_A",
  "detector": "gaussian",
  "latest_version": "v2",
  "versions": ["v1", "v2"]
}
```

---

## Rollout Phases (Layer-by-Layer)

### Phase 1 — Domain layer

**`IsolationForestDetector` reference implementation:**

```python
class IsolationForestDetector:
    def fit(self, data: TimeSeries) -> "IsolationForestDetector":
        values = [[p.value] for p in data.data]
        self._clf = IsolationForest(
            n_estimators=100,
            contamination="auto",
            random_state=42,
        ).fit(values)
        scores = self._clf.score_samples(values)
        self._threshold = float(np.percentile(scores, 10))
        return self

    def predict(self, data_point: DataPoint) -> bool:
        score = float(self._clf.score_samples([[data_point.value]])[0])
        return score < self._threshold
```

`model_params` in `metadata.json`:
```json
{"n_estimators": 100, "contamination": "auto", "score_threshold": -0.412}
```

- `DetectorType` literal, `DetectorProtocol`, `IsolationForestDetector`
- detector-aware metadata schema (`model_params`, optional `mean`/`std`)
- `detector` field in `FitResponse` and `PredictResponse`
- `UnsupportedDetectorError`, `VersionNotFoundForDetectorError`
- `UNSUPPORTED_DETECTOR` and `VERSION_NOT_FOUND_FOR_DETECTOR` in `error_handlers.py`
- `scikit-learn` in `pyproject.toml`
- **Unit tests:** `IsolationForestDetector` fit + predict; new exception types

### Phase 2 — Repository layer

- detector-scoped paths and index methods
- `schema_version: "1"` written to `index.json` on every save (forward-compatibility marker, no validation behavior)
- **Unit tests:** repository with detector scoping; save/load/index per `(series_id, detector)`; version resolution scoped per detector

### Phase 3 — Service layer

- `ModelService.train` and `predict` with `detector` param
- version resolution per `(series_id, detector)`
- **Unit tests:** service dispatch to correct detector; default gaussian behavior

### Phase 4 — API layer

- `?detector=` on `/fit`, `/predict`, `/models*`
- response schemas updated with `detector` field
- **Integration tests:**
  - gaussian default behavior unchanged (regression)
  - isolation_forest train + predict success
  - both detectors coexist for same `series_id` from v1 (clean layout, no migration)
  - invalid detector → `422 UNSUPPORTED_DETECTOR`
  - `?detector=` on `/models*` returns correct scoped data

### Phase 5 — Docs + scripts + benchmark

- `ARCHITECTURE.md` — detector contract, storage contract
- `README.md` — detector query examples and compatibility notes
- FastAPI live docs — `?detector=` and normalized error responses
- `scripts/examples/fit_request.sh` and `predict_request.sh` — add `?detector=` optional usage
- `scripts/benchmark.py` — optional: cover isolation_forest path alongside gaussian default

---

## Exit Criteria

- All pre-existing default flows pass unchanged (gaussian path).
- Detector-scoped versioning works without ambiguity.
- Detector-scoped layout adopted from v1 — no legacy artifacts, no migration required.
- New detector errors are normalized and documented.
- Full suite passes with all detector scenarios (no storage migration required or tested).

---

## Validation Commands

```bash
pytest -v
```
