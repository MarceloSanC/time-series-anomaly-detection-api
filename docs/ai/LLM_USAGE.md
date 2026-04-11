# LLM_USAGE.md — HOW TO USE GPT-CODEX AND CLAUDE EFFECTIVELY

This file defines the working protocol for AI-assisted development on this project.
Read before every coding session.

---

## ROLE SPLIT

### Use Claude Sonnet 4.6 for:
- Reviewing generated code against `docs/ai/skills.md` and `docs/project/ARCHITECTURE.md`
- Discussing architectural trade-offs when two approaches seem valid
- Generating test cases, especially edge cases and negative cases for `ValidationService`
- Writing documentation: README sections, docstrings, error messages
- Identifying boundary violations between layers
- Debugging non-obvious issues (concurrency, atomic writes, serialization)

### Use GPT-5.3-codex for:
- Generating boilerplate: route handlers given a request/response schema
- Implementing defined algorithms: `LockManager`, `MetricsService`, `ModelRepository`
- Generating project scaffolding: `pyproject.toml`, `Makefile`, `Dockerfile`
- Implementing the benchmark script
- Generating sample request scripts

---

## SESSION STARTUP PROTOCOL

At the start of every session with either model:

1. Paste `docs/ai/skills.md` in full as the first message (system context)
2. State which stage of the roadmap you are on
3. State which layer you are working in (domain / services / repository / api / utils / tests)
4. State which files already exist and are complete
5. State what you need generated or reviewed

Example session opener:
```
[Paste docs/ai/skills.md here]

---

Context:
- Stage: 2
- Layer: api/routes
- Already complete: domain/schemas.py, domain/models.py, services/model_service.py, repository/model_repository.py, utils/concurrency.py
- Task: Generate app/api/routes/fit.py for POST /fit/{series_id}
  - Request body: Trainining.TrainData from docs/context/openapi_spec.yaml
  - Success response: Trainining.TrainResponse from docs/context/openapi_spec.yaml
  - Calls: model_service.train(series_id, mapped_time_series)
  - Must map fields: timestamps/values -> TimeSeries.data; n_samples -> points_used; prediction timestamp string -> internal int
  - Error handling: SeriesValidationError → 400, unexpected → 500
  - Must include: request timing, structured logging with request_id
```

---

## CODE REVIEW PROTOCOL (Claude)

After GPT-codex generates any file, ask Claude to review it with this prompt:

```
Review this generated code against docs/ai/skills.md. Check for:
1. Type hints on all function signatures
2. Correct layer boundaries (no violations)
3. Error handling completeness
4. Logging present where required
5. No hardcoded values that should be in config
6. Correct Pydantic v2 usage
7. Thread safety for any shared state
8. Any pattern that violates docs/ai/skills.md

[Paste generated code here]
```

---

## TEST GENERATION PROTOCOL (Claude)

For any service or function:

```
Generate pytest tests for this function following docs/ai/skills.md testing rules:
- Unit tests: no filesystem, no HTTP, mock repository
- At least 1 happy path
- At least 1 error path per possible exception
- For ValidationService: 1 passing case + 1 rejection case per rule

Function to test:
[Paste function signature + docstring here]

Existing test fixtures in conftest.py:
[Paste conftest.py if it exists]
```

---

## WHAT NEVER TO ACCEPT FROM LLMs

Reject any generated code that:

- Uses `pickle` directly (must use `joblib`)
- Puts business logic inside route handlers
- Accesses the filesystem from the `api/` layer
- Uses a global variable for model storage (must go through repository)
- Has a bare `except:` without logging and re-raise
- Hardcodes paths (must use `config.STORAGE_PATH`)
- Uses `asyncio.Lock` or `run_in_executor` for training (use `threading.Lock`)
- Returns a raw Python exception to the API client
- Suggests adding a dependency not in the approved stack
- Starts implementing an enhancement before the MVP checklist is complete

If any of these appear, reject the output and regenerate with an explicit constraint.

---

## PROMPTING TIPS

### For route handlers:
Always provide the exact request and response Pydantic schemas. Do not let the model invent schemas.

### For repository:
Always specify the exact filesystem layout. Show the `index.json` structure. Do not let the model choose its own storage format.

### For validation:
List the exact rules in order. Specify the error code and HTTP status for each. Do not let the model invent new rules.

### For tests:
Provide the function signature and docstring. Specify what already exists in `conftest.py`. Ask for specific edge cases by name if needed.

### For Docker:
Specify that the base image must be `python:3.11-slim`, that a non-root user must be created, and that `./storage` must be mounted as a volume.

---

## DEBUGGING PROTOCOL

When something is broken:

1. Reproduce the bug with the smallest possible input
2. Identify which layer the failure originates in
3. Ask Claude to explain the failure and suggest root causes before generating a fix
4. Ask GPT-codex to implement the fix Claude proposes
5. Write a regression test that catches the exact failure before applying the fix

Do not ask GPT-codex to "fix" something without first understanding what is wrong.

---

## SCOPE MANAGEMENT

If either model suggests:
- A new endpoint not in the spec or enhancement list → reject
- A new dependency → evaluate cost/benefit explicitly, default to reject
- Changing the anomaly detection algorithm → reject (out of scope)
- An abstraction layer not in the project structure → reject unless Stage 5 polish

When in doubt, ask: "Does this help deliver the MVP faster, or does it add complexity?"
If it adds complexity and the MVP is not done: reject.
