# SCALABILITY REFACTOR PLAN — MULTI-DETECTOR EVOLUTION

## Purpose

This document isolates the scalability-heavy refactor from the regular technical roadmap.
It defines how each module is impacted before implementing multi-detector runtime support.

Why separate:
- scope is cross-cutting (domain, services, repository, API, docs, tests, migration)
- changes affect persistence contracts and runtime resolution semantics
- safer execution as a dedicated implementation program with explicit rollout phases

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

## Architectural Decisions (Must Be Frozen Before Code)

1. Detector namespace is explicit and required for non-default behavior.
2. Default detector remains `gaussian`.
3. Version resolution is detector-scoped only (no cross-detector fallback).
4. Storage/index is detector-scoped.
5. Legacy artifacts are interpreted as `gaussian` and migrated on-read.

---

## Module Impact Matrix

### Domain Layer

- `app/domain/models.py`
  - add detector interface/contract
  - add `IsolationForestDetector` with parity (`fit`, `predict`)
- `app/domain/schemas.py`
  - add detector type/enum
  - add detector-aware metadata schema (common + detector-specific fields)
- `app/domain/exceptions.py`
  - add detector/version namespace errors:
    - `UnsupportedDetectorError`
    - `VersionNotFoundForDetectorError`

### Repository Layer

- `app/repository/model_repository.py`
  - detector-aware paths:
    - `storage/{series_id}/{detector}/{version}/...`
    - `storage/{series_id}/{detector}/index.json`
  - detector-aware save/load/index/version methods
  - on-read legacy migration hooks (atomic + idempotent)

### Service Layer

- `app/services/model_service.py`
  - extend train/predict with `detector` selection
  - resolve version inside selected detector namespace
  - keep default gaussian behavior when detector not provided
  - ensure list/info flows have defined detector behavior

### API Layer

- `app/api/routes/fit.py`
  - optional `?detector=` with default gaussian
- `app/api/routes/predict.py`
  - optional `?detector=` with default gaussian
- `app/api/routes/models.py`
  - define whether no-detector view is gaussian-only or detector-parameterized
- `app/api/error_handlers.py`
  - map new detector errors to normalized payload:
    - `UNSUPPORTED_DETECTOR` -> 422
    - `VERSION_NOT_FOUND_FOR_DETECTOR` -> 404

### Configuration and Dependencies

- `pyproject.toml`
  - add `scikit-learn`
- `app/config.py`
  - optional detector defaults/config knobs if needed

### Docs

- `docs/project/ARCHITECTURE.md`
  - detector contract, storage contract, migration-on-read behavior
- `README.md`
  - detector query examples and compatibility notes
- `docs/project/API_RESPONSES.md`
  - new detector-related error codes

### Scripts

- `scripts/compare_detectors.py`
  - benchmark script after runtime multi-detector implementation lands

### Tests

- `tests/unit/*`
  - detector model unit tests
  - repository migration/namespace unit tests
  - service resolution unit tests
- `tests/integration/*`
  - fit/predict gaussian default regression
  - fit/predict isolation_forest success
  - mismatch `(detector, version)` handling
  - legacy artifact migration-on-read

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

## Rollout Phases

1. **Phase 1 — Contract freeze**
   - finalize detector, runtime, storage, and error contracts
2. **Phase 2 — Repository + migration**
   - implement detector-scoped storage and on-read migration
3. **Phase 3 — Service + API wiring**
   - add detector routing in train/predict and error mappings
4. **Phase 4 — Tests + docs**
   - regression + migration + detector behavior tests
   - documentation updates
5. **Phase 5 — Benchmark and evidence**
   - compare detector behavior and record tradeoffs

---

## Exit Criteria

- All pre-existing default flows still pass unchanged (gaussian path).
- Detector-scoped versioning works without ambiguity.
- Legacy artifacts migrate safely on-read.
- New detector errors are normalized and documented.
- Full suite passes with detector and migration scenarios.

---

## Validation Commands (Program-Level)

```bash
pytest -v
.venv/bin/python scripts/compare_detectors.py
```

