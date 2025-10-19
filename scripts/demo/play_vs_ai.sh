#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_FILE="$(mktemp -t chessbrain-play-XXXX.log)"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  rm -f "${LOG_FILE}"
}
trap cleanup EXIT

export PYTHONPATH="${ROOT_DIR}"

if [[ -z "${DATABASE_URL:-}" ]]; then
  SQLITE_PATH="${ROOT_DIR}/.chessbrain-play.sqlite"
  export DATABASE_URL="sqlite+pysqlite:///${SQLITE_PATH}"
elif [[ "${DATABASE_URL}" == sqlite+pysqlite:///* ]]; then
  SQLITE_PATH="${DATABASE_URL#sqlite+pysqlite:///}"
else
  SQLITE_PATH=""
fi

if [[ -n "${SQLITE_PATH}" ]]; then
  mkdir -p "$(dirname "${SQLITE_PATH}")"
  touch "${SQLITE_PATH}"
fi

export MODEL_CHECKPOINT_DIR="${MODEL_CHECKPOINT_DIR:-${ROOT_DIR}/models/checkpoints}"
export TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR:-${ROOT_DIR}/tb_logs}"
export FLASK_APP="src.chessbrain.interface.http.app:create_app"
export FLASK_ENV="${FLASK_ENV:-production}"

MODEL_CANDIDATE="${ACTIVE_MODEL_PATH:-${ROOT_DIR}/models/checkpoints/c3ade8cf-step185-20251019171110.pt}"
if [[ ! -f "${MODEL_CANDIDATE}" ]]; then
  ARCHIVE_FALLBACK="${ROOT_DIR}/models/checkpoints.archive/c3ade8cf-step185-20251019171110.pt"
  if [[ -f "${ARCHIVE_FALLBACK}" ]]; then
    MODEL_CANDIDATE="${ARCHIVE_FALLBACK}"
  fi
fi
if [[ ! -f "${MODEL_CANDIDATE}" ]]; then
  echo "Model checkpoint not found. Set ACTIVE_MODEL_PATH to a valid .pt file." >&2
  exit 1
fi
export ACTIVE_MODEL_PATH="${MODEL_CANDIDATE}"

echo "[chessbrain] Starting Flask server..." >&2
python -m flask run --port 5001 --no-reload >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

for i in {1..30}; do
  if curl -sf "http://localhost:5001/healthz" >/dev/null; then
    break
  fi
  sleep 1
  if [[ $i -eq 30 ]]; then
    echo "Server failed to start. Logs:" >&2
    cat "${LOG_FILE}" >&2
    exit 1
  fi
done

echo "[chessbrain] Creating match session..." >&2
CREATE_PAYLOAD='{"playerColor":"white","difficulty":"deterministic"}'
SESSION_JSON="$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -d "${CREATE_PAYLOAD}" \
  "http://localhost:5001/api/v1/sessions")"

SESSION_ID="$(python -c 'import json,sys; print(json.load(sys.stdin)["id"])' <<<"${SESSION_JSON}")"
echo "[chessbrain] Session ${SESSION_ID} created." >&2

echo "[chessbrain] Playing opening move e2e4..." >&2
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"uci":"e2e4"}' \
  "http://localhost:5001/api/v1/sessions/${SESSION_ID}/moves" > /dev/null

echo "[chessbrain] Fetching latest state..." >&2
FINAL_STATE="$(curl -s "http://localhost:5001/api/v1/sessions/${SESSION_ID}")"

python - <<'PY'
import json, sys

state = json.loads(sys.stdin.read())
moves = state.get("moves", [])
print("=== ChessBrain Match Demo ===")
print(f"Session: {state.get('id')}")
print(f"Status : {state.get('status')}")
print(f"Model  : {state.get('activeModelVersion')}")
print("Moves  :")
for idx, move in enumerate(moves, start=1):
    actor = "You" if move.get("actor") == "human" else "AI "
    san = move.get("san")
    print(f"  {idx:02d}. {actor} -> {san}")
PY
