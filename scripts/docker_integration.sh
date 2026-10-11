#!/usr/bin/env bash
# Build is done by the caller. This starts the image and checks the contract.
set -euo pipefail

IMAGE="${1:-placer-lookalike:local}"
PORT="${2:-18080}"
NAME="placer-itest-$$"
BASE="http://127.0.0.1:${PORT}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker run -d --name "$NAME" \
  -p "${PORT}:8000" \
  -e PLACER_RATE_LIMIT=0 \
  -e PYTHONUNBUFFERED=1 \
  "$IMAGE" >/dev/null

ok=0
for _ in $(seq 1 60); do
  if curl -sf "$BASE/health" >/dev/null; then
    ok=1
    break
  fi
  sleep 1
done
if [[ "$ok" -ne 1 ]]; then
  echo "container never became healthy" >&2
  docker logs "$NAME" >&2 || true
  exit 1
fi

python3 "$ROOT/scripts/smoke_health.py" "$BASE"

code="$(curl -s -o /tmp/placer-predict.json -w '%{http_code}' \
  -H 'Content-Type: application/json' \
  -d '{"Th":12,"Ce":66,"La":38,"P":700,"U":3.2,"Au":0.004,"As":1.5,"Ti":520,"Fe":2.7,"Zr":96,"Y":18,"fe_unit":"wt_pct"}' \
  "$BASE/predict")"
if [[ "$code" != 200 ]]; then
  echo "expected /predict 200, got $code" >&2
  cat /tmp/placer-predict.json >&2
  exit 1
fi

code="$(curl -s -o /tmp/placer-bad.json -w '%{http_code}' \
  -H 'Content-Type: application/json' \
  -d '{"Th":"hot","Ce":66,"La":38,"P":700,"U":3.2,"Au":0.004,"As":1.5,"Ti":520,"Fe":2.7,"Zr":96,"Y":18,"fe_unit":"wt_pct"}' \
  "$BASE/predict")"
if [[ "$code" != 422 ]]; then
  echo "expected /predict 422, got $code" >&2
  cat /tmp/placer-bad.json >&2
  exit 1
fi

python3 - "$BASE" <<'PY'
import json, sys, urllib.request
base = sys.argv[1].rstrip("/")
with urllib.request.urlopen(base + "/metrics", timeout=10) as resp:
    body = json.loads(resp.read().decode())
assert body["model_version"], body
assert body["request_count"] >= 1, body
print("metrics ok", body["model_version"], "requests", body["request_count"])
PY

docker logs "$NAME" 2>&1 | python3 -c '
import json, sys
chemistry = {
    "Th", "Ce", "La", "P", "U", "Au", "As", "Ti", "Fe", "Zr", "Y",
    "fe_unit", "lon", "lat",
}
required = {
    "timestamp", "model_version", "endpoint", "status",
    "latency_ms", "feature_validation",
}
saw_predict = False
for raw in sys.stdin:
    line = raw.strip()
    if not line.startswith("{"):
        continue
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    leaked = chemistry.intersection(event)
    if leaked:
        raise SystemExit("chemistry leaked into logs: %s" % sorted(leaked))
    if event.get("endpoint") == "/predict":
        missing = required.difference(event)
        if missing:
            raise SystemExit("structured log missing %s: %s" % (sorted(missing), event))
        saw_predict = True
if not saw_predict:
    raise SystemExit("no structured /predict log line from the container")
print("structured logs ok (no chemistry fields)")
'

echo "container integration ok"
