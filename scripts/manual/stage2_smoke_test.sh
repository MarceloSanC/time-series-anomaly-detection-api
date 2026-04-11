#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

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

assert_fit_success() {
  local response="$1"
  local expected_series="$2"
  local expected_version="$3"
  echo "${response}" | python3 -c '
import json
import sys

p = json.loads(sys.stdin.read())
if "series_id" not in p:
    raise SystemExit(f"unexpected fit payload: {p}")
assert p["series_id"] == sys.argv[1]
assert p["version"] == sys.argv[2]
assert p["points_used"] >= 30
print(f"  - {sys.argv[1]} trained:", p)
' "${expected_series}" "${expected_version}"
}

echo "[1/4] Training 3 different series_id values..."

train_sensor_a_v1="$(curl -sS -X POST "${BASE_URL}/fit/sensor_A" \
  -H "Content-Type: application/json" \
  -d "$(build_series_payload 1700000001 10.0)")"
assert_fit_success "${train_sensor_a_v1}" "sensor_A" "v1"

train_sensor_b_v1="$(curl -sS -X POST "${BASE_URL}/fit/sensor_B" \
  -H "Content-Type: application/json" \
  -d "$(build_series_payload 1700000101 20.0)")"
assert_fit_success "${train_sensor_b_v1}" "sensor_B" "v1"

train_sensor_c_v1="$(curl -sS -X POST "${BASE_URL}/fit/sensor_C" \
  -H "Content-Type: application/json" \
  -d "$(build_series_payload 1700000201 30.0)")"
assert_fit_success "${train_sensor_c_v1}" "sensor_C" "v1"

echo "[2/4] Predicting on each series..."

predict_sensor_a="$(curl -sS -X POST "${BASE_URL}/predict/sensor_A" \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"1700000301","value":99.0}')"
echo "${predict_sensor_a}" | python3 -c '
import json,sys
p=json.loads(sys.stdin.read())
assert "anomaly" in p and isinstance(p["anomaly"], bool)
assert p["model_version"]=="v1"
print("  - sensor_A prediction:", p)
'

predict_sensor_b="$(curl -sS -X POST "${BASE_URL}/predict/sensor_B" \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"1700000302","value":19.5}')"
echo "${predict_sensor_b}" | python3 -c '
import json,sys
p=json.loads(sys.stdin.read())
assert "anomaly" in p and isinstance(p["anomaly"], bool)
assert p["model_version"]=="v1"
print("  - sensor_B prediction:", p)
'

predict_sensor_c="$(curl -sS -X POST "${BASE_URL}/predict/sensor_C" \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"1700000303","value":35.0}')"
echo "${predict_sensor_c}" | python3 -c '
import json,sys
p=json.loads(sys.stdin.read())
assert "anomaly" in p and isinstance(p["anomaly"], bool)
assert p["model_version"]=="v1"
print("  - sensor_C prediction:", p)
'

echo "[3/4] Verifying version increment on retrain..."

train_sensor_a_v2="$(curl -sS -X POST "${BASE_URL}/fit/sensor_A" \
  -H "Content-Type: application/json" \
  -d "$(build_series_payload 1700000401 13.0)")"
assert_fit_success "${train_sensor_a_v2}" "sensor_A" "v2"

predict_sensor_a_latest="$(curl -sS -X POST "${BASE_URL}/predict/sensor_A" \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"1700000501","value":25.0}')"
echo "${predict_sensor_a_latest}" | python3 -c '
import json,sys
p=json.loads(sys.stdin.read())
assert p["model_version"]=="v2"
print("  - sensor_A latest predict (default version):", p)
'

echo "[4/4] Verifying /healthcheck contract payload..."

healthcheck_payload="$(curl -sS "${BASE_URL}/healthcheck")"
echo "${healthcheck_payload}" | python3 -c '
import json,sys
p=json.loads(sys.stdin.read())
assert isinstance(p.get("series_trained"), int)
assert isinstance(p.get("inference_latency_ms"), dict)
assert isinstance(p.get("training_latency_ms"), dict)
assert {"avg", "p95"} <= set(p["inference_latency_ms"].keys())
assert {"avg", "p95"} <= set(p["training_latency_ms"].keys())
print("  - healthcheck:", p)
'

echo
echo "Stage 2 manual smoke test passed."
