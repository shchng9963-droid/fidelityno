#!/bin/bash
# Ablation runs for supplementary S4
# Variants: noaux, aux05, gaussian
# Seeds: 0, 1, 2
# 9 runs total, 2 GPUs

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python}"

mkdir -p checkpoints/collision_ablation
mkdir -p results_prxq/ablation/training_logs

declare -a JOBS=()
JOBS+=("noaux 0 model.aux=False train.aux_weight=0.0")
JOBS+=("noaux 1 model.aux=False train.aux_weight=0.0")
JOBS+=("noaux 2 model.aux=False train.aux_weight=0.0")
JOBS+=("aux05 0 train.aux_weight=0.5")
JOBS+=("aux05 1 train.aux_weight=0.5")
JOBS+=("aux05 2 train.aux_weight=0.5")
JOBS+=("gauss 0 model.head_type=gaussian")
JOBS+=("gauss 1 model.head_type=gaussian")
JOBS+=("gauss 2 model.head_type=gaussian")

# 2 GPUs, dispatch round-robin
GPU=0
for spec in "${JOBS[@]}"; do
    read -r tag seed extras <<< "$spec"
    LOG="results_prxq/ablation/training_logs/${tag}_seed${seed}.log"
    CKPT_NAME="${tag}_seed${seed}.pt"
    CMD="CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} train.py model.name=fidelityno seed=${seed} \
        data.train=data/collision/train.npz data.val=data/collision/calib.npz \
        train.epochs=80 train.batch_size=256 train.patience=20 \
        train.ckpt_dir=checkpoints/collision_ablation \
        train.ckpt_name=${CKPT_NAME} \
        ${extras}"
    echo "[launch] gpu${GPU} ${tag}_seed${seed} -> ${LOG}"
    echo "         $ ${CMD}" >> "${LOG}"
    eval "${CMD} > \"${LOG}\" 2>&1 &"
    GPU=$(( (GPU + 1) % 2 ))
    # Wait if 2 jobs already running on each GPU
    while [ $(jobs -r | wc -l) -ge 2 ]; do
        sleep 5
    done
done

wait
echo "[summary] all ablation training runs finished"
