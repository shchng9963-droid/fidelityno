#!/usr/bin/env bash
# Train all models on N GPUs in parallel — one job per GPU at a time.
# Usage: EPOCHS=80 BATCH=512 NUM_GPUS=2 bash scripts/train_all.sh
#
# Robust dispatch: tracks per-GPU PIDs in a plain array, no command-substitution
# capture of stdout (which broke the previous launcher and ran 35 jobs at once
# on 2 GPUs).
set -uo pipefail
cd /home/wangshuchang/fidelityno

ENV=fidelityno
EPOCHS=${EPOCHS:-80}
BATCH=${BATCH:-512}
NUM_GPUS=${NUM_GPUS:-2}
HEADLINE=${HEADLINE:-fidelityno_large}   # which config is the "main" FidelityNO
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}     # e.g. "train.lr=2e-4"
CKPT_DIR=checkpoints
LOG_DIR=logs
mkdir -p $CKPT_DIR $LOG_DIR

# Build job list. Order matters: put the most expensive jobs first
# so the long pole runs in parallel with the rest.
declare -a JOBS=()
for seed in 0 1 2 3 4; do
  for model in fidelityno_large fidelityno bidir gnn generic_gnn mlp deepsets; do
    if [[ "$model" == "fidelityno_large" ]]; then
      model_cfg="fidelityno_large"
    else
      model_cfg="fidelityno"
    fi

    ckpt_name="${model}_seed${seed}.pt"
    log_file="${LOG_DIR}/train_${model}_seed${seed}.log"

    # Skip only if log claims convergence below the uniform-predictor plateau.
    if [ -s "$log_file" ] && grep -q "best_val_pinball" "$log_file" 2>/dev/null; then
      val=$(grep -oP 'best_val_pinball=\K[\d.]+' "$log_file" | tail -1)
      if [ -n "$val" ] && awk "BEGIN{exit !($val < 0.05)}" 2>/dev/null; then
        echo "SKIP $model seed=$seed (already converged: best_val_pinball=$val)"
        continue
      else
        echo "RETRAIN $model seed=$seed (last best_val_pinball=$val >= 0.05)"
      fi
    fi

    rm -f "$log_file" "$CKPT_DIR/$ckpt_name"
    JOBS+=("${model}|${model_cfg}|${seed}|${ckpt_name}|${log_file}")
  done
done

echo "Total jobs: ${#JOBS[@]}    GPUs: $NUM_GPUS"

# Per-GPU PID tracking.
declare -a GPU_PIDS
for ((g=0; g<NUM_GPUS; g++)); do GPU_PIDS[$g]=0; done

start_job() {
  local job_str="$1"
  local gpu="$2"
  IFS='|' read -r model model_cfg seed ckpt_name log_file <<< "$job_str"

  local overrides="model=$model_cfg model.name=$model seed=$seed"
  overrides+=" train.ckpt_name=$ckpt_name"
  overrides+=" train.epochs=$EPOCHS train.batch_size=$BATCH"
  if [[ "$model" == "bidir" ]]; then overrides+=" model.causal=false"; fi
  if [[ -n "$EXTRA_OVERRIDES" ]]; then overrides+=" $EXTRA_OVERRIDES"; fi

  echo "[$(date +%H:%M:%S)] [GPU$gpu] start $model seed=$seed -> $log_file"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 \
    nohup conda run -n $ENV python train.py $overrides \
    >"$log_file" 2>&1 &
  GPU_PIDS[$gpu]=$!
}

# Find a free GPU slot (PID==0 or process gone). Returns the index.
find_free_gpu() {
  while true; do
    for ((g=0; g<NUM_GPUS; g++)); do
      local pid=${GPU_PIDS[$g]}
      if [[ $pid -eq 0 ]] || ! kill -0 "$pid" 2>/dev/null; then
        # Reap if there was a real PID
        if [[ $pid -ne 0 ]]; then wait "$pid" 2>/dev/null || true; fi
        echo "$g"
        return
      fi
    done
    sleep 10
  done
}

job_i=0
for job_str in "${JOBS[@]}"; do
  gpu=$(find_free_gpu)
  start_job "$job_str" "$gpu"
  job_i=$((job_i+1))
  echo "[$(date +%H:%M:%S)] launched $job_i/${#JOBS[@]} on GPU$gpu (pid=${GPU_PIDS[$gpu]})"
  sleep 3   # let CUDA init before sharing the bus
done

# Final wait
echo "[$(date +%H:%M:%S)] all jobs dispatched, waiting for last $NUM_GPUS to finish..."
for ((g=0; g<NUM_GPUS; g++)); do
  pid=${GPU_PIDS[$g]}
  if [[ $pid -ne 0 ]] && kill -0 "$pid" 2>/dev/null; then
    wait "$pid" 2>/dev/null || true
  fi
done

echo "=== All training done ==="
for seed in 0 1 2 3 4; do
  for model in fidelityno_large fidelityno bidir gnn generic_gnn mlp deepsets; do
    log="${LOG_DIR}/train_${model}_seed${seed}.log"
    if [ -f "$log" ] && grep -q "best_val_pinball" "$log" 2>/dev/null; then
      val=$(grep -oP 'best_val_pinball=\K[\d.]+' "$log" | tail -1)
      printf "  %-22s seed=%d  best_val_pinball=%s\n" "$model" "$seed" "$val"
    else
      printf "  %-22s seed=%d  FAILED\n" "$model" "$seed"
    fi
  done
done
