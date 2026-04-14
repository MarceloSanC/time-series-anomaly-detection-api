#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
STORAGE_PATH="${STORAGE_PATH:-./storage}"
TIMESTAMP=$(date +%s)
SERIES_A="smoke_${TIMESTAMP}_A"
SERIES_B="smoke_${TIMESTAMP}_B"
SERIES_C="smoke_${TIMESTAMP}_C"

cleanup() {
  rm -rf "${STORAGE_PATH}/${SERIES_A}" \
         "${STORAGE_PATH}/${SERIES_B}" \
         "${STORAGE_PATH}/${SERIES_C}"
}
trap cleanup EXIT

build_series_payload() {
  local start_ts="$1"
  local start_value="$2"
  python3 -c '
import json
import sys

start_ts = int(sys.argv[1])
start_value = float(sys.argv[2])
timestamps = [start_ts + i for i in range(30)]
values = [start_value + (i * 0.2) for i in range(30)]
print(json.dumps({"timestamps": timestamps, "values": values}))
' "$start_ts" "$start_value"
}

assert_fit_response() {
  local response="$1"
  local expected_series="$2"
  echo "${response}" | python3 -c '
import json, sys

p = json.loads(sys.stdin.read())
if "series_id" not in p:
    raise SystemExit(f"unexpected fit payload: {p}")
assert p["series_id"] == sys.argv[1], f"expected series_id={sys.argv[1]}, got {p}"
assert p["version"].startswith("v"), f"version must start with v, got {p}"
assert int(p["version"][1:]) >= 1, f"version must be >= v1, got {p}"
assert p["points_used"] >= 30, f"expected points_used >= 30, got {p}"
print(f"  - {sys.argv[1]} trained:", p)
' "${expected_series}"
}

extract_version() {
  echo "$1" | python3 -c 'import json, sys; print(json.loads(sys.stdin.read())["version"])'
}

assert_version_incremented() {
  local v_before="$1"
  local v_after="$2"
  python3 -c '
import sys
v1, v2 = sys.argv[1], sys.argv[2]
assert int(v2[1:]) == int(v1[1:]) + 1, f"expected version increment: {v1} -> {v2}"
print(f"  - version incremented correctly: {v1} -> {v2}")
' "${v_before}" "${v_after}"
}

echo "[1/4] Training 3 isolated smoke series..."

train_a="$(curl -sS -X POST "${BASE_URL}/fit/${SERIES_A}" \
  -H "Content-Type: application/json" \
  -d "$(build_series_payload 1700000001 10.0)")"
assert_fit_response "${train_a}" "${SERIES_A}"
version_a=$(extract_version "${train_a}")

train_b="$(curl -sS -X POST "${BASE_URL}/fit/${SERIES_B}" \
  -H "Content-Type: application/json" \
  -d "$(build_series_payload 1700000101 20.0)")"
assert_fit_response "${train_b}" "${SERIES_B}"
version_b=$(extract_version "${train_b}")

train_c="$(curl -sS -X POST "${BASE_URL}/fit/${SERIES_C}" \
  -H "Content-Type: application/json" \
  -d "$(build_series_payload 1700000201 30.0)")"
assert_fit_response "${train_c}" "${SERIES_C}"
version_c=$(extract_version "${train_c}")

echo "[2/4] Predicting on each series..."

predict_a="$(curl -sS -X POST "${BASE_URL}/predict/${SERIES_A}" \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"1700000301","value":99.0}')"
echo "${predict_a}" | python3 -c '
import json, sys
p = json.loads(sys.stdin.read())
assert "anomaly" in p and isinstance(p["anomaly"], bool), f"missing anomaly field: {p}"
assert p["model_version"] == sys.argv[1], f"expected version={sys.argv[1]}, got {p}"
print("  - prediction A:", p)
' "${version_a}"

predict_b="$(curl -sS -X POST "${BASE_URL}/predict/${SERIES_B}" \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"1700000302","value":19.5}')"
echo "${predict_b}" | python3 -c '
import json, sys
p = json.loads(sys.stdin.read())
assert "anomaly" in p and isinstance(p["anomaly"], bool), f"missing anomaly field: {p}"
assert p["model_version"] == sys.argv[1], f"expected version={sys.argv[1]}, got {p}"
print("  - prediction B:", p)
' "${version_b}"

predict_c="$(curl -sS -X POST "${BASE_URL}/predict/${SERIES_C}" \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"1700000303","value":35.0}')"
echo "${predict_c}" | python3 -c '
import json, sys
p = json.loads(sys.stdin.read())
assert "anomaly" in p and isinstance(p["anomaly"], bool), f"missing anomaly field: {p}"
assert p["model_version"] == sys.argv[1], f"expected version={sys.argv[1]}, got {p}"
print("  - prediction C:", p)
' "${version_c}"

echo "[3/4] Verifying version increment on retrain..."

retrain_a="$(curl -sS -X POST "${BASE_URL}/fit/${SERIES_A}" \
  -H "Content-Type: application/json" \
  -d "$(build_series_payload 1700000401 13.0)")"
assert_fit_response "${retrain_a}" "${SERIES_A}"
version_a_new=$(extract_version "${retrain_a}")
assert_version_incremented "${version_a}" "${version_a_new}"

predict_a_latest="$(curl -sS -X POST "${BASE_URL}/predict/${SERIES_A}" \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"1700000501","value":25.0}')"
echo "${predict_a_latest}" | python3 -c '
import json, sys
p = json.loads(sys.stdin.read())
assert p["model_version"] == sys.argv[1], f"expected latest version={sys.argv[1]}, got {p}"
print("  - prediction A after retrain (latest):", p)
' "${version_a_new}"

echo "[4/4] Verifying /healthcheck contract payload..."

healthcheck_payload="$(curl -sS "${BASE_URL}/healthcheck")"
echo "${healthcheck_payload}" | python3 -c '
import json, sys
p = json.loads(sys.stdin.read())
assert isinstance(p.get("series_trained"), int)
assert isinstance(p.get("inference_latency_ms"), dict)
assert isinstance(p.get("training_latency_ms"), dict)
assert {"avg", "p95"} <= set(p["inference_latency_ms"].keys())
assert {"avg", "p95"} <= set(p["training_latency_ms"].keys())
print("  - healthcheck:", p)
'

echo
echo "Stage 2 manual smoke test passed."
