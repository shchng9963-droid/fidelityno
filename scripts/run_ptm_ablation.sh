#!/bin/bash
# PTM-encoder ablation training: 3 seeds on PTM-converted collision data
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python}"

mkdir -p checkpoints/collision_ablation
mkdir -p results_prxq/ablation/training_logs

declare -a JOBS=("ptm 0" "ptm 1" "ptm 2")
GPU=0
for spec in "${JOBS[@]}"; do
    read -r tag seed <<< "$spec"
    LOG="results_prxq/ablation/training_logs/${tag}_seed${seed}.log"
    CMD="CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} train.py model.name=fidelityno seed=${seed} \
        data.train=data/collision_ptm/train.npz data.val=data/collision_ptm/calib.npz \
        train.epochs=80 train.batch_size=256 train.patience=20 \
        train.ckpt_dir=checkpoints/collision_ablation \
        train.ckpt_name=${tag}_seed${seed}.pt"
    echo "[launch] gpu${GPU} ${tag}_seed${seed} -> ${LOG}"
    eval "${CMD} > \"${LOG}\" 2>&1 &"
    GPU=$(( (GPU + 1) % 2 ))
    while [ $(jobs -r | wc -l) -ge 2 ]; do
        sleep 5
    done
done

wait
echo "[summary] PTM ablation training finished"
