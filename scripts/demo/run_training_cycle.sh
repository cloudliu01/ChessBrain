#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-$REPO_ROOT}"
export DATABASE_URL="${DATABASE_URL:-sqlite+pysqlite:///$REPO_ROOT/.chessbrain-training.sqlite}"
export MODEL_CHECKPOINT_DIR="${MODEL_CHECKPOINT_DIR:-$REPO_ROOT/models/checkpoints}"
export TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR:-$REPO_ROOT/data/tensorboard}"

AMP_FLAG=""
if [[ "${NO_AMP:-0}" == "1" ]]; then
  AMP_FLAG="--no-amp"
fi

echo "Starting ChessBrain training cycle..."
cmd=(
  python -m src.chessbrain.interface.cli.train
  --episodes "${EPISODES:-4}"
  --batch-size "${BATCH_SIZE:-1024}"
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-2}"
  --exploration-rate "${EXPLORATION_RATE:-0.1}"
  --seed "${SEED:-7}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-1}"
  --mcts-simulations "${MCTS_SIMULATIONS:-32}"
  --mcts-cpuct "${MCTS_CPUCT:-1.5}"
)

if [[ -n "${AMP_FLAG}" ]]; then
  cmd+=("${AMP_FLAG}")
fi

"${cmd[@]}" | while IFS= read -r line; do
  echo "$line"
done

LATEST_JOB_DIR=$(find "${TENSORBOARD_LOG_DIR}" -maxdepth 1 -type d -name "[0-9a-f]*" -print | sort | tail -n 1 || true)
if [[ -n "${LATEST_JOB_DIR}" ]]; then
  TOTAL_EPISODES="${EPISODES:-4}"
  LAST_EPISODE=$(jq -r '.["episode_index"]' "${LATEST_JOB_DIR}/metrics.jsonl" 2>/dev/null | tail -n 1 || echo 0)
  PERCENT=0
  if [[ "${TOTAL_EPISODES}" -gt 0 ]]; then
    PERCENT=$(( LAST_EPISODE * 100 / TOTAL_EPISODES ))
  fi
  echo "Training progress: ${LAST_EPISODE}/${TOTAL_EPISODES} episodes (${PERCENT}%)."
  echo "Metrics log: ${LATEST_JOB_DIR}/metrics.jsonl"
fi

LATEST_CHECKPOINT=$(find "${MODEL_CHECKPOINT_DIR}" -type f -name '*.pt' -print | sort | tail -n 1 || true)
if [[ -n "${LATEST_CHECKPOINT}" ]]; then
  echo "Latest checkpoint: ${LATEST_CHECKPOINT}"
fi
