#!/usr/bin/env bash
# ============================================================================
# FidelityNO — Family-OOD ROBUSTNESS SWEEP
# 5 models x 4 hold-out families x 3 seeds = 60 training runs.
#
# Holdouts: amplitude_damping, phase_damping, depolarizing, lindblad
# (the original main run already covers `pauli` as the held-out family.)
#
# Models: fidelityno (fnoT), gnn (fnoG), generic_gnn (path-MP / no Choi enc),
#          mlp, deepsets
#
# Outputs:
#   data/family_ood/holdout_<F>/{train,id_test,length_ood,family_ood,
#                                train_split,val_split}.npz
#   checkpoints/family_ood/<model>_holdout_<F>_seed<S>.pt
#   results/family_ood_sweep/<model>_holdout_<F>_seed<S>.csv
#   logs/family_ood/{gen,train,eval}_<...>.log
# ============================================================================
set -euo pipefail

ENV=/home/wangshuchang/miniforge3/envs/fidelityno/bin
ROOT=/home/wangshuchang/fidelityno
cd "$ROOT"
export WANDB_MODE=offline
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

EPOCHS=${EPOCHS:-80}
BATCH=${BATCH:-512}
N_TRAIN=${N_TRAIN:-200000}
N_TEST=${N_TEST:-10000}
SEEDS=${SEEDS:-"0 1 2"}
HOLDOUTS=${HOLDOUTS:-"amplitude_damping phase_damping depolarizing lindblad"}
MODELS=${MODELS:-"fidelityno gnn generic_gnn mlp deepsets"}
GPU_COUNT=${GPU_COUNT:-2}

mkdir -p data/family_ood logs/family_ood results/family_ood_sweep checkpoints/family_ood

# Model-specific overrides — must match scripts/run_all_baselines.sh so that
# parameter budgets are identical to the headline ID/length-OOD tables.
declare -A CFG
CFG["fidelityno"]="model=fidelityno"
CFG["gnn"]="model.name=gnn model.d_model=256 model.layers=6 model.head_type=quantile"
CFG["generic_gnn"]="model.name=generic_gnn model.d_model=256 model.layers=3 model.head_type=quantile"
CFG["mlp"]="model.name=mlp model.d_model=256 model.head_type=quantile"
CFG["deepsets"]="model.name=deepsets model.d_model=768 model.head_type=quantile"

echo "============================================================"
echo "Family-OOD sweep configuration"
echo "  EPOCHS=$EPOCHS  BATCH=$BATCH  N_TRAIN=$N_TRAIN  N_TEST=$N_TEST"
echo "  HOLDOUTS=$HOLDOUTS"
echo "  MODELS=$MODELS"
echo "  SEEDS=$SEEDS"
echo "  GPUS=$GPU_COUNT"
echo "============================================================"

# =============================================================================
# Phase 1: Data generation (CPU-parallel, one process per holdout)
# =============================================================================
PHASE=${PHASE:-all}      # 'data', 'train', or 'all'

gen_one_holdout() {
    local h=$1
    local outdir="data/family_ood/holdout_${h}"
    local manifest="${outdir}/manifest.json"
    if [ -f "$manifest" ] && [ -f "${outdir}/train_split.npz" ] && [ -f "${outdir}/val_split.npz" ]; then
        echo "  [data] holdout=${h}: cached, skipping"
        return 0
    fi
    mkdir -p "$outdir"
    local glog="logs/family_ood/gen_${h}.log"
    echo "  [data] holdout=${h} -> ${outdir} (log: ${glog})"
    "$ENV/python" scripts/gen_data.py \
        --outdir "$outdir" \
        --n-train "$N_TRAIN" --n-test "$N_TEST" \
        --seed 0 \
        --holdout-family "$h" \
        --max-len 48 \
        --representation choi_hermitian \
        > "$glog" 2>&1
    "$ENV/python" scripts/split_data_holdout.py --indir "$outdir" >> "$glog" 2>&1
}

if [ "$PHASE" = "data" ] || [ "$PHASE" = "all" ]; then
    echo "[Phase 1] Generating data for $(echo $HOLDOUTS | wc -w) holdouts in parallel ..."
    DATA_PIDS=()
    for h in $HOLDOUTS; do
        gen_one_holdout "$h" &
        DATA_PIDS+=($!)
    done
    fail=0
    for pid in "${DATA_PIDS[@]}"; do
        wait "$pid" || { echo "  [data] PID $pid failed"; fail=1; }
    done
    if [ "$fail" -ne 0 ]; then
        echo "[Phase 1] One or more data jobs failed; see logs/family_ood/gen_*.log"
        exit 1
    fi
    echo "[Phase 1] Done."
fi

# =============================================================================
# Phase 2: Training (GPU-parallel, 1 job per GPU)
# =============================================================================
declare -A GPU_PID
declare -A GPU_TAG
for g in $(seq 0 $((GPU_COUNT - 1))); do GPU_PID[$g]=0; GPU_TAG[$g]=""; done

run_train_eval() {
    local model=$1 holdout=$2 seed=$3 gpu=$4
    local datadir="data/family_ood/holdout_${holdout}"
    local tag="${model}_holdout_${holdout}_seed${seed}"
    local ckpt="checkpoints/family_ood/${tag}.pt"
    local log="logs/family_ood/train_${tag}.log"
    local elog="logs/family_ood/eval_${tag}.log"
    local out="results/family_ood_sweep/${tag}.csv"
    local cfg="${CFG[$model]}"

    if [ -f "$out" ]; then
        echo "  [skip] $tag (already evaluated -> $out)"
        return 0
    fi
    (
        set -e
        export CUDA_VISIBLE_DEVICES=$gpu
        cd "$ROOT"
        echo "  [GPU $gpu] train: $tag"
        "$ENV/python" train.py \
            $cfg seed=$seed \
            data.train="${datadir}/train_split.npz" \
            data.val="${datadir}/val_split.npz" \
            train.epochs="$EPOCHS" train.batch_size="$BATCH" device=cuda \
            train.ckpt_dir=checkpoints/family_ood \
            train.ckpt_name="${tag}.pt" \
            > "$log" 2>&1
        echo "  [GPU $gpu] eval:  $tag"
        "$ENV/python" eval.py \
            --ckpt "$ckpt" \
            --data-dir "$datadir" \
            --out "$out" \
            > "$elog" 2>&1
    )
}

if [ "$PHASE" = "train" ] || [ "$PHASE" = "all" ]; then
    echo "[Phase 2] Training $((`echo $MODELS|wc -w` * `echo $HOLDOUTS|wc -w` * `echo $SEEDS|wc -w`)) jobs on $GPU_COUNT GPUs ..."

    JOBS=()
    for h in $HOLDOUTS; do
        for s in $SEEDS; do
            for m in $MODELS; do
                JOBS+=("$m|$h|$s")
            done
        done
    done

    # Simple round-robin scheduler across GPUs.
    for job in "${JOBS[@]}"; do
        IFS='|' read -r m h s <<< "$job"
        # Wait for a free GPU slot.
        while :; do
            for g in $(seq 0 $((GPU_COUNT - 1))); do
                pid=${GPU_PID[$g]}
                if [ "$pid" = "0" ] || ! kill -0 "$pid" 2>/dev/null; then
                    if [ "$pid" != "0" ]; then
                        wait "$pid" || echo "  [GPU $g] job '${GPU_TAG[$g]}' failed"
                        GPU_PID[$g]=0; GPU_TAG[$g]=""
                    fi
                    run_train_eval "$m" "$h" "$s" "$g" &
                    GPU_PID[$g]=$!
                    GPU_TAG[$g]="${m}_${h}_s${s}"
                    break 2
                fi
            done
            sleep 5
        done
    done

    # Wait for stragglers.
    for g in $(seq 0 $((GPU_COUNT - 1))); do
        pid=${GPU_PID[$g]}
        if [ "$pid" != "0" ]; then
            wait "$pid" || echo "  [GPU $g] final job '${GPU_TAG[$g]}' failed"
        fi
    done
    echo "[Phase 2] Done."
fi

echo "============================================================"
echo "Sweep complete. Aggregate with scripts/aggregate_family_ood.py"
echo "============================================================"
