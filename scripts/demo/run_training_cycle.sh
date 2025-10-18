#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-$REPO_ROOT}"
export DATABASE_URL="${DATABASE_URL:-sqlite+pysqlite:///$REPO_ROOT/.chessbrain-training.sqlite}"
export MODEL_CHECKPOINT_DIR="${MODEL_CHECKPOINT_DIR:-$REPO_ROOT/models/checkpoints}"
export TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR:-$REPO_ROOT/data/tensorboard}"

echo "Starting ChessBrain training cycle..."
python -m src.chessbrain.interface.cli.train \
  --episodes "${EPISODES:-4}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-2}" \
  --exploration-rate "${EXPLORATION_RATE:-0.1}" \
  --seed "${SEED:-7}"

echo "Training cycle complete. Checkpoints stored in ${MODEL_CHECKPOINT_DIR}."
