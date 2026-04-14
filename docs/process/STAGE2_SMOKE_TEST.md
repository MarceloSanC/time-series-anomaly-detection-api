# Stage 2 Manual Smoke Test

This document records the manual smoke test procedure required by Stage 2 Task 7.

## Goal

Validate manually that the API satisfies:

1. Train 3 different `series_id` values
2. Predict on each series
3. Verify version increments on retrain
4. Verify `GET /healthcheck` returns contract-compatible payload

## Prerequisites

- API running locally (for example with `docker compose up -d`)
- `.env` configured

## Command

From repository root:

```bash
UID=$(id -u) GID=$(id -g) docker compose up -d --build
./scripts/manual/stage2_smoke_test.sh
```

If API is exposed on a different URL:

```bash
BASE_URL="http://localhost:8000" ./scripts/manual/stage2_smoke_test.sh
```

## Pass Criteria

The script exits with code `0` and prints:

```text
Stage 2 manual smoke test passed.
```

Any contract violation or unexpected response causes an immediate failure due to `set -euo pipefail` and Python assertions.
