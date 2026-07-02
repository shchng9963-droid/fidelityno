#!/usr/bin/env bash
# Run real-hardware eval on the device-regime checkpoints.
# Skips models with missing checkpoints (fail loud only at the end).

set -e
set -u
set -o pipefail

cd "$(dirname "$0")/.."

CKPT_ROOT="checkpoints/device_regime"
OUT_ROOT="results_prxq/real_hardware/nn_models_device"
DATA_ROOT="data/real_hardware"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[error] $DATA_ROOT missing — generate real-hardware splits first."
  exit 1
fi

mkdir -p "$OUT_ROOT"

ckpts=()
for model in fidelityno fidelityno_large mlp deepsets bidir; do
  for seed in 0 1 2 3 4; do
    p="$CKPT_ROOT/${model}_seed${seed}.pt"
    if [[ -f "$p" ]]; then
      ckpts+=("$p")
    else
      echo "[warn] missing $p"
    fi
  done
done

if [[ ${#ckpts[@]} -eq 0 ]]; then
  echo "[error] no device-regime checkpoints in $CKPT_ROOT yet."
  exit 2
fi

echo "[run] eval on ${#ckpts[@]} ckpts -> $OUT_ROOT"
python scripts/eval_real_hardware.py \
  --ckpts "${ckpts[@]}" \
  --data-root "$DATA_ROOT" \
  --out-root "$OUT_ROOT" \
  --device cpu

# Build the unified comparison table
python scripts/build_real_hardware_v2_summary.py
