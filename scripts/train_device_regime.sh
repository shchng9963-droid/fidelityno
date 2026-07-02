#!/usr/bin/env bash
# Retrain FidelityNO + key baselines on the device-regime data
# distribution (PRXQ track P0.1b).
#
# Usage:
#   bash scripts/train_device_regime.sh
#
# Output: checkpoints/device_regime/<model>_seed<S>.pt

set -e
set -u
set -o pipefail

cd "$(dirname "$0")/.."

DATA_DIR="data/device_regime"
CKPT_DIR="checkpoints/device_regime"
mkdir -p "$CKPT_DIR"

if [[ ! -f "$DATA_DIR/train.npz" ]]; then
  echo "[error] $DATA_DIR/train.npz not found; run gen_data.py --regime device first."
  exit 1
fi

PY=python
EPOCHS=${DEVICE_EPOCHS:-80}
BATCH=${DEVICE_BATCH:-512}

# 1) FidelityNO + FidelityNO-Large need the model config (uses depth/heads)
for model in fidelityno fidelityno_large; do
  for seed in 0 1 2 3 4; do
    ckpt_name="${model}_seed${seed}.pt"
    if [[ -f "$CKPT_DIR/$ckpt_name" ]]; then
      echo "[skip] $CKPT_DIR/$ckpt_name exists"
      continue
    fi
    echo "[train] device-regime $model seed=$seed"
    $PY train.py \
      model=${model} \
      seed=${seed} \
      data.train="${DATA_DIR}/train.npz" \
      data.val="${DATA_DIR}/calib.npz" \
      train.epochs=${EPOCHS} \
      train.batch_size=${BATCH} \
      train.ckpt_dir="${CKPT_DIR}" \
      train.ckpt_name="${ckpt_name}" \
      train.patience=25
  done
done

# 2) Baselines (mlp, deepsets, bidir) - just override model.name; the
# default fidelityno.yaml model fields drive d_model and head_type.
for model in deepsets mlp bidir; do
  for seed in 0 1 2 3 4; do
    ckpt_name="${model}_seed${seed}.pt"
    if [[ -f "$CKPT_DIR/$ckpt_name" ]]; then
      echo "[skip] $CKPT_DIR/$ckpt_name exists"
      continue
    fi
    echo "[train] device-regime $model seed=$seed"
    $PY train.py \
      model.name=${model} \
      seed=${seed} \
      data.train="${DATA_DIR}/train.npz" \
      data.val="${DATA_DIR}/calib.npz" \
      train.epochs=${EPOCHS} \
      train.batch_size=${BATCH} \
      train.ckpt_dir="${CKPT_DIR}" \
      train.ckpt_name="${ckpt_name}" \
      train.patience=25
  done
done

echo "[done] device-regime retraining complete:"
ls -la "$CKPT_DIR"/*.pt | wc -l
echo "checkpoints in $CKPT_DIR/"
