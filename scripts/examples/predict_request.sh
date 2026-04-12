#!/usr/bin/env bash

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
SERIES_ID="${SERIES_ID:-sensor_XYZ}"
VERSION_QUERY="${VERSION_QUERY:-}"

if [[ -n "${VERSION_QUERY}" ]]; then
  URL="${BASE_URL}/predict/${SERIES_ID}?version=${VERSION_QUERY}"
else
  URL="${BASE_URL}/predict/${SERIES_ID}"
fi

curl --fail-with-body -sS -X POST "${URL}" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "1700000100",
    "value": 99.0
  }'

echo
