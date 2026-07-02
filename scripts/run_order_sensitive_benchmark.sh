#!/usr/bin/env bash
set -euo pipefail

cd /home/wangshuchang/fidelityno

ENV_BIN=/home/wangshuchang/miniforge3/envs/fidelityno/bin
export WANDB_MODE=${WANDB_MODE:-offline}

BENCH=two_qubit_order_sensitive
SEEDS=${SEEDS:-"0 1 2 3 4"}
MODELS=${MODELS:-"fidelityno gnn generic_gnn mlp deepsets bidir"}
GPU_IDS=${GPU_IDS:-"0 1"}
N_TRAIN=${N_TRAIN:-10000}
N_CALIB=${N_CALIB:-2000}
N_TEST=${N_TEST:-2000}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-128}
DEVICE=${DEVICE:-cuda}
MC_BUDGETS=${MC_BUDGETS:-10,100,1000}
MC_MAX_EVAL=${MC_MAX_EVAL:-1000}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}

BENCH_ROOT=${BENCH_ROOT:-data/benchmarks}
DATA_DIR="$BENCH_ROOT/$BENCH"
OUT_ROOT=${OUT_ROOT:-results/benchmarks}
OUT_DIR="$OUT_ROOT/$BENCH"
CKPT_DIR=${CKPT_DIR:-checkpoints/order_sensitive_${RUN_TAG}}
JOB_LOG_DIR=${JOB_LOG_DIR:-logs/order_sensitive_${RUN_TAG}}

mkdir -p logs "$CKPT_DIR" "$OUT_DIR" "$JOB_LOG_DIR"

mkdir -p "$DATA_DIR"

echo "[$(date '+%F %T')] generate benchmark dataset bench=$BENCH train=$N_TRAIN calib=$N_CALIB test=$N_TEST"
$ENV_BIN/python scripts/gen_data.py \
  --outdir "$DATA_DIR" \
  --n-train "$N_TRAIN" \
  --n-calib "$N_CALIB" \
  --n-test "$N_TEST" \
  --seed 0 \
  --dim 4 \
  --family order_sensitive \
  --holdout-family correlated_dephasing \
  --representation choi_hermitian \
  --train-lengths 8,16 \
  --id-lengths 8,16 \
  --length-ood-lengths 24,32,48 \
  --family-ood-lengths 8,16,24

echo "[$(date '+%F %T')] benchmark=$BENCH analytical baselines"
$ENV_BIN/python scripts/eval_analytic.py --data-dir "$DATA_DIR" --out "$OUT_DIR/analytic.csv"

echo "[$(date '+%F %T')] benchmark=$BENCH MC budgets=$MC_BUDGETS max_eval=$MC_MAX_EVAL"
$ENV_BIN/python scripts/eval_mc.py --data-dir "$DATA_DIR" --out "$OUT_DIR/mc.csv" --budgets "$MC_BUDGETS" --max-eval "$MC_MAX_EVAL"

declare -a GPU_ARRAY
read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}
declare -a JOBS=()

for seed in $SEEDS; do
  for model in $MODELS; do
    JOBS+=("${model}|${seed}")
  done
done

run_job() {
  local model="$1"
  local seed="$2"
  local gpu="$3"
  local ckpt="${model}_${BENCH}_seed${seed}.pt"
  local job_log="$JOB_LOG_DIR/${model}_seed${seed}.log"
  local train_overrides=(
    "model.name=$model"
    "seed=$seed"
    "data.train=$DATA_DIR/train.npz"
    "data.val=$DATA_DIR/id_test.npz"
    "train.epochs=$EPOCHS"
    "train.batch_size=$BATCH"
    "train.ckpt_dir=$CKPT_DIR"
    "train.ckpt_name=$ckpt"
    "device=$DEVICE"
  )
  if [[ "$model" == "bidir" ]]; then
    train_overrides+=("model.causal=false")
  fi

  echo "[$(date '+%F %T')] benchmark=$BENCH start model=$model seed=$seed gpu=$gpu log=$job_log"
  (
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$ENV_BIN/python" train.py "${train_overrides[@]}"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$ENV_BIN/python" eval.py --ckpt "$CKPT_DIR/$ckpt" --data-dir "$DATA_DIR" --out "$OUT_DIR/${model}_seed${seed}.csv"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$ENV_BIN/python" scripts/eval_calibrated.py --ckpt "$CKPT_DIR/$ckpt" --data-dir "$DATA_DIR" --out "$OUT_DIR/${model}_seed${seed}_calibrated.csv"
  ) >"$job_log" 2>&1
}

declare -a GPU_PIDS
for ((slot=0; slot<NUM_GPUS; slot++)); do GPU_PIDS[$slot]=0; done

start_job() {
  local job_str="$1"
  local slot="$2"
  local gpu="${GPU_ARRAY[$slot]}"
  IFS='|' read -r model seed <<< "$job_str"
  run_job "$model" "$seed" "$gpu" &
  GPU_PIDS[$slot]=$!
}

find_free_slot() {
  while true; do
    for ((slot=0; slot<NUM_GPUS; slot++)); do
      local pid=${GPU_PIDS[$slot]}
      if [[ $pid -eq 0 ]] || ! kill -0 "$pid" 2>/dev/null; then
        if [[ $pid -ne 0 ]]; then wait "$pid" 2>/dev/null || true; fi
        echo "$slot"
        return
      fi
    done
    sleep 10
  done
}

job_i=0
for job_str in "${JOBS[@]}"; do
  slot=$(find_free_slot)
  start_job "$job_str" "$slot"
  job_i=$((job_i+1))
  echo "[$(date '+%F %T')] dispatched $job_i/${#JOBS[@]} on gpu=${GPU_ARRAY[$slot]} pid=${GPU_PIDS[$slot]}"
  sleep 2
done

echo "[$(date '+%F %T')] all training/eval jobs dispatched; waiting for completion"
for ((slot=0; slot<NUM_GPUS; slot++)); do
  pid=${GPU_PIDS[$slot]}
  if [[ $pid -ne 0 ]] && kill -0 "$pid" 2>/dev/null; then
    wait "$pid" 2>/dev/null || true
  fi
done

$ENV_BIN/python - "$OUT_DIR" <<'PY'
from pathlib import Path
import sys
import pandas as pd

out = Path(sys.argv[1])
main = [out / 'analytic.csv', out / 'mc.csv'] + [f for f in sorted(out.glob('*_seed*.csv')) if 'calibrated' not in f.name]
pd.concat([pd.read_csv(f) for f in main if f.exists()], ignore_index=True).to_csv(out / 'summary.csv', index=False)
cal = sorted(out.glob('*_calibrated.csv'))
if cal:
    pd.concat([pd.read_csv(f) for f in cal], ignore_index=True).to_csv(out / 'calibrated_summary.csv', index=False)
print('wrote', out / 'summary.csv')
PY

$ENV_BIN/python scripts/build_order_sensitive_summary.py
echo "[$(date '+%F %T')] order-sensitive benchmark complete"