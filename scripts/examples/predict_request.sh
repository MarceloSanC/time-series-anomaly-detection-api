#!/usr/bin/env bash

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
SERIES_ID="${SERIES_ID:-sensor_XYZ}"
VERSION_QUERY="${VERSION_QUERY:-}"
DETECTOR="${DETECTOR:-}"

URL="${BASE_URL}/predict/${SERIES_ID}"
HAS_QUERY=false

if [[ -n "${VERSION_QUERY}" ]]; then
  URL="${URL}?version=${VERSION_QUERY}"
  HAS_QUERY=true
fi

if [[ -n "${DETECTOR}" ]]; then
  if [[ "${HAS_QUERY}" == "true" ]]; then
    URL="${URL}&detector=${DETECTOR}"
  else
    URL="${URL}?detector=${DETECTOR}"
  fi
fi

curl --fail-with-body -sS -X POST "${URL}" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "1700000100",
    "value": 99.0
  }'

echo
