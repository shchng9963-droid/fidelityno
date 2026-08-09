#!/usr/bin/env bash
# C2: Train all architectures on L<=4 only (extreme length-OOD probe).
# Outputs go to checkpoints/length_extreme/  and logs to logs/length_extreme/
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV=fidelityno
EPOCHS=${EPOCHS:-40}            # smaller dataset, fewer epochs needed
BATCH=${BATCH:-512}
NUM_GPUS=${NUM_GPUS:-2}
SEEDS=${SEEDS:-"0 1 2"}
MODELS=${MODELS:-"fidelityno bidir gnn generic_gnn mlp deepsets"}

CKPT_DIR=checkpoints/length_extreme
LOG_DIR=logs/length_extreme
mkdir -p $CKPT_DIR $LOG_DIR

DATA_TRAIN=data/length_extreme/train.npz
DATA_VAL=data/length_extreme/val.npz

declare -a JOBS=()
for seed in $SEEDS; do
  for model in $MODELS; do
    model_cfg="fidelityno"      # all share the same model size (small)
    ckpt_name="${model}_seed${seed}.pt"
    log_file="${LOG_DIR}/train_${model}_seed${seed}.log"
    if [ -s "$log_file" ] && grep -q "best_val_pinball" "$log_file" 2>/dev/null; then
      echo "SKIP $model seed=$seed (already trained)"; continue
    fi
    rm -f "$log_file" "$CKPT_DIR/$ckpt_name"
    JOBS+=("${model}|${model_cfg}|${seed}|${ckpt_name}|${log_file}")
  done
done

echo "Total jobs: ${#JOBS[@]}    GPUs: $NUM_GPUS"

declare -a GPU_PIDS
for ((g=0; g<NUM_GPUS; g++)); do GPU_PIDS[$g]=0; done

start_job() {
  local job_str="$1" gpu="$2"
  IFS='|' read -r model model_cfg seed ckpt_name log_file <<< "$job_str"

  local overrides="model=$model_cfg model.name=$model seed=$seed"
  overrides+=" data.train=$DATA_TRAIN data.val=$DATA_VAL"
  overrides+=" train.ckpt_name=$ckpt_name train.ckpt_dir=$CKPT_DIR"
  overrides+=" train.epochs=$EPOCHS train.batch_size=$BATCH"
  overrides+=" train.curriculum=false"
  if [[ "$model" == "bidir" ]]; then overrides+=" model.causal=false"; fi

  echo "[$(date +%H:%M:%S)] [GPU$gpu] start $model seed=$seed"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 \
    nohup conda run -n $ENV python train.py $overrides \
    >"$log_file" 2>&1 &
  GPU_PIDS[$gpu]=$!
}

find_free_gpu() {
  while true; do
    for ((g=0; g<NUM_GPUS; g++)); do
      local pid=${GPU_PIDS[$g]}
      if [[ $pid -eq 0 ]] || ! kill -0 "$pid" 2>/dev/null; then
        if [[ $pid -ne 0 ]]; then wait "$pid" 2>/dev/null || true; fi
        echo "$g"; return
      fi
    done
    sleep 5
  done
}

for job_str in "${JOBS[@]}"; do
  gpu=$(find_free_gpu)
  start_job "$job_str" "$gpu"
  sleep 3
done

# Wait for all
for ((g=0; g<NUM_GPUS; g++)); do
  pid=${GPU_PIDS[$g]}
  [[ $pid -ne 0 ]] && wait "$pid" 2>/dev/null || true
done
echo "ALL DONE at $(date)"
