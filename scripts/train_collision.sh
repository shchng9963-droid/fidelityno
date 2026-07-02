#!/usr/bin/env bash
# Train FidelityNO + key baselines on the non-Markovian collision dataset
# (PRXQ P1.1).
#
# This is the *positive* counter-example: per-channel marginals see
# (rho_S, |+>) joint while the true sequence dynamics retain bath
# correlation eta in [0, 0.7] (train) / [0.85, 0.99] (family OOD).
# The analytic product bound provably fails here.
#
# Usage:
#   bash scripts/train_collision.sh
#   DEVICE_GPUS=0,1 bash scripts/train_collision.sh

set -e -u -o pipefail
cd "$(dirname "$0")/.."

DATA_DIR="data/collision"
CKPT_DIR="checkpoints/collision"
LOG_DIR="results_prxq/collision/training_logs"
mkdir -p "$CKPT_DIR" "$LOG_DIR"

if [[ ! -f "$DATA_DIR/train.npz" ]]; then
  echo "[error] $DATA_DIR/train.npz missing; run scripts/gen_collision_data.py first."
  exit 1
fi

EPOCHS=${COLL_EPOCHS:-80}
BATCH=${COLL_BATCH:-256}

# Driver: reuse train_device_regime_parallel.py with overrides
DEVICE_GPUS=${DEVICE_GPUS:-0,1} \
DEVICE_EPOCHS=${EPOCHS} \
DEVICE_BATCH=${BATCH} \
DEVICE_PATIENCE=20 \
COLL_DATA_DIR="${DATA_DIR}" \
COLL_CKPT_DIR="${CKPT_DIR}" \
COLL_LOG_DIR="${LOG_DIR}" \
python scripts/train_collision_parallel.py
