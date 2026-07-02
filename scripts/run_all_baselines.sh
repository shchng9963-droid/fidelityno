#!/usr/bin/env bash
# ============================================================================
# FidelityNO — Full experiment pipeline
# Usage:
#   bash scripts/run_all_baselines.sh           # full run, 100 epochs, 200k samples
#   EPOCHS=3 N_TRAIN=1000 N_TEST=300 bash scripts/run_all_baselines.sh   # quick smoke test
# ============================================================================
set -euo pipefail

ENV=/home/wangshuchang/miniforge3/envs/fidelityno/bin
cd /home/wangshuchang/fidelityno
export WANDB_MODE=offline

# --- Defaults (override via env vars) ---
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-128}
N_TRAIN=${N_TRAIN:-200000}
N_TEST=${N_TEST:-10000}
MC_BUDGETS=${MC_BUDGETS:-10,100,1000}
HOLDOUT=${HOLDOUT_FAMILY:-pauli}

echo "============================================"
echo "FidelityNO experiment pipeline"
echo "  EPOCHS=$EPOCHS  BATCH=$BATCH"
echo "  N_TRAIN=$N_TRAIN  N_TEST=$N_TEST"
echo "  HOLDOUT_FAMILY=$HOLDOUT"
echo "============================================"

# --- Step 1: Generate data if needed ---
echo "[Step 1] Checking / generating data..."
$ENV/python - <<'PY'
import json, numpy as np, os, subprocess, sys
regen = os.environ.get('FORCE_REGEN_DATA','0') == '1' or not os.path.exists('data/train.npz')
if not regen:
    try:
        n = len(np.load('data/train.npz', allow_pickle=True)['y'])
        target = int(os.environ.get('N_TRAIN', '200000'))
        regen = n < target * 0.9  # regen if substantially fewer
        if os.path.exists('data/manifest.json'):
            manifest = json.load(open('data/manifest.json'))
            regen = regen or not isinstance(manifest, dict) or 'holdout_family' not in manifest
        else:
            regen = True
    except Exception:
        regen = True
if regen:
    print(f"Generating data: N_TRAIN={os.environ.get('N_TRAIN','200000')}, N_TEST={os.environ.get('N_TEST','10000')}")
    subprocess.check_call([
        '/home/wangshuchang/miniforge3/envs/fidelityno/bin/python',
        'scripts/gen_data.py',
        '--n-train', os.environ.get('N_TRAIN','200000'),
        '--n-test', os.environ.get('N_TEST','10000'),
        '--seed', '0',
        '--holdout-family', os.environ.get('HOLDOUT_FAMILY','pauli'),
    ])
else:
    print(f"Data exists with {n} train samples, skipping regen.")
PY

# --- Step 2: Analytical baselines (B1, B2) ---
echo "[Step 2] Analytical baselines..."
$ENV/python scripts/eval_analytic.py --out results/analytic.csv

# --- Step 3: Monte Carlo baseline (B3) ---
echo "[Step 3] Monte Carlo baseline..."
$ENV/python scripts/eval_mc.py --out results/mc.csv --budgets $MC_BUDGETS ${MC_MAX_EVAL:+--max-eval $MC_MAX_EVAL}

# --- Step 4: Neural model training (5 seeds × 7 models, parallel on 2 GPUs) ---
echo "[Step 4] Training neural models..."
rm -rf results/*_seed*.csv
mkdir -p checkpoints results

# Models and their config overrides
declare -A MODEL_CFGS
MODEL_CFGS["fidelityno"]="model=fidelityno"
MODEL_CFGS["fidelityno_large"]="model=fidelityno_large"
MODEL_CFGS["bidir"]="model=fidelityno model.name=bidir model.causal=false"
MODEL_CFGS["mlp"]="model.name=mlp model.d_model=256 model.head_type=quantile"
MODEL_CFGS["deepsets"]="model.name=deepsets model.d_model=768 model.head_type=quantile"
MODEL_CFGS["gnn"]="model.name=gnn model.d_model=256 model.layers=6 model.head_type=quantile"
MODEL_CFGS["generic_gnn"]="model.name=generic_gnn model.d_model=256 model.layers=3 model.head_type=quantile"

# Run: 2 GPUs, distribute jobs
PIDS=()
JOB_QUEUE=()

for seed in 0 1 2 3 4; do
  for model in fidelityno fidelityno_large bidir mlp deepsets gnn generic_gnn; do
    JOB_QUEUE+=("$seed|$model")
  done
done

GPU_COUNT=2
running=0
declare -A GPU_PIDS

run_job() {
    local seed=$1 model=$2 gpu=$3
    local cfg="${MODEL_CFGS[$model]}"
    echo "  [GPU $gpu] Training $model seed=$seed"
    CUDA_VISIBLE_DEVICES=$gpu $ENV/python train.py \
        $cfg seed=$seed \
        train.epochs=$EPOCHS train.batch_size=$BATCH device=cuda \
        train.ckpt_dir=checkpoints train.ckpt_name=${model}_seed${seed}.pt \
        > logs/train_${model}_seed${seed}.log 2>&1 &
    echo $!
}

mkdir -p logs

# Simple parallel scheduler: keep up to 2 jobs running
for job in "${JOB_QUEUE[@]}"; do
    IFS='|' read -r seed model <<< "$job"

    # Wait if both GPUs are busy
    while [ $running -ge $GPU_COUNT ]; do
        for g in $(seq 0 $((GPU_COUNT-1))); do
            pid=${GPU_PIDS[$g]:-0}
            if [ "$pid" != "0" ] && ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" || echo "WARNING: GPU $g job (PID $pid) failed"
                GPU_PIDS[$g]=0
                running=$((running - 1))
            fi
        done
        if [ $running -ge $GPU_COUNT ]; then
            sleep 5
        fi
    done

    # Find free GPU
    for g in $(seq 0 $((GPU_COUNT-1))); do
        pid=${GPU_PIDS[$g]:-0}
        if [ "$pid" = "0" ] || ! kill -0 "$pid" 2>/dev/null; then
            GPU_PIDS[$g]=$(run_job $seed $model $g)
            running=$((running + 1))
            break
        fi
    done
done

# Wait for all remaining
for g in $(seq 0 $((GPU_COUNT-1))); do
    pid=${GPU_PIDS[$g]:-0}
    if [ "$pid" != "0" ]; then
        wait "$pid" || echo "WARNING: GPU $g final job failed"
    fi
done

echo "  All training done."

# --- Step 5: Evaluation ---
echo "[Step 5] Evaluating all checkpoints..."
for seed in 0 1 2 3 4; do
    for model in fidelityno fidelityno_large bidir mlp deepsets gnn generic_gnn; do
        ckpt="checkpoints/${model}_seed${seed}.pt"
        if [ -f "$ckpt" ]; then
            $ENV/python eval.py --ckpt "$ckpt" --out "results/${model}_seed${seed}.csv"
        else
            echo "  WARNING: $ckpt not found, skipping eval"
        fi
    done
done

# --- Step 6: Aggregate results ---
echo "[Step 6] Aggregating results..."
$ENV/python - <<'PY'
from pathlib import Path
import pandas as pd
files = [Path('results/analytic.csv'), Path('results/mc.csv')]
files += sorted(Path('results').glob('*_seed*.csv'))
dfs = [pd.read_csv(f) for f in files if f.exists()]
if dfs:
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv('results/summary.csv', index=False)
    print(f'Wrote results/summary.csv with {len(df)} rows')
    # Print summary table
    summary = df.groupby(['model','split']).agg({
        'mae': ['mean','std'],
        'pinball': ['mean','std'],
        'ece': ['mean','std'],
    }).round(5)
    print(summary.to_string())
PY

# --- Step 7: Make figures ---
echo "[Step 7] Generating figures..."
$ENV/python scripts/make_figures.py

echo "============================================"
echo "DONE. Results in results/summary.csv"
echo "Figures in results/figs/"
echo "============================================"
