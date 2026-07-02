#!/bin/bash
# Reproduce all PRXQ-track experiments end-to-end.
#
# Pre-reqs:
#   - conda env `fidelityno` activated
#   - /home/wangshuchang/fidelityno_prxq is the working directory
#   - 2x NVIDIA GPU (or single-GPU mode by setting DEVICE_GPUS=0)
#
# Total budget at default settings: ~5-7 hours on 2x RTX 5090.
#
# Run: bash scripts/run_all_prxq.sh [phase]
#   phase: empty -> run everything; "data", "train", "eval", "figures"
#
set -euo pipefail
cd "$(dirname "$0")/.."

PHASE="${1:-all}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-512}"
N_TRAIN="${N_TRAIN:-80000}"
N_TEST="${N_TEST:-8000}"

mkdir -p data results_prxq logs

run_phase() {
  case "$1" in
    data|all)
      echo "=== [phase data] generating datasets ==="
      python scripts/gen_device_regime_data.py --outdir data/device_regime \
        --n-train "${N_TRAIN}" --n-calib "${N_TEST}" --n-test "${N_TEST}" --seed 0
      python scripts/gen_collision_data.py --outdir data/collision \
        --n-train "${N_TRAIN}" --n-calib "${N_TEST}" --n-test "${N_TEST}" --seed 0
      ;;
  esac
  case "$1" in
    train|all)
      echo "=== [phase train] training NN models (5 seeds x 4 archs x 2 datasets) ==="
      DEVICE_GPUS="${DEVICE_GPUS:-0,1}" DEVICE_EPOCHS="${EPOCHS}" \
        DEVICE_BATCH="${BATCH}" DEVICE_PATIENCE=20 \
        python scripts/train_device_regime_parallel.py \
        2>&1 | tee logs/train_device_regime.log
      DEVICE_GPUS="${DEVICE_GPUS:-0,1}" DEVICE_EPOCHS="${EPOCHS}" \
        DEVICE_BATCH="${BATCH}" DEVICE_PATIENCE=20 \
        python scripts/train_collision_parallel.py \
        2>&1 | tee logs/train_collision.log
      ;;
  esac
  case "$1" in
    eval|all)
      echo "=== [phase eval] running all baselines ==="
      # Device regime (real hardware): NN models on real-hardware test data
      bash scripts/eval_device_regime_real_hw.sh
      # Collision: analytic + MC + DFE + per-seed NN + recalibrated
      python scripts/eval_analytic.py --data-dir data/collision \
        --out results_prxq/collision/analytic.csv
      python scripts/eval_mc.py --data-dir data/collision \
        --out results_prxq/collision/mc.csv \
        --budgets 10,100,1000 --max-eval 1024
      for s in id_test length_ood family_ood; do
        TQDM_DISABLE=1 python scripts/eval_dfe.py \
          --data data/collision/${s}.npz \
          --pauli-budgets 10,30,100,300,1000 --M 200 --n-eval 1024 \
          --out results_prxq/collision/dfe_${s}.csv
      done
      python -c "
import pandas as pd, glob
dfs=[pd.read_csv(f).assign(split=f.split('_',-1)[-1].replace('.csv','')) for f in glob.glob('results_prxq/collision/dfe_*.csv')]
pd.concat(dfs).to_csv('results_prxq/collision/dfe.csv', index=False)
"
      for m in fidelityno mlp deepsets bidir; do for s in 0 1 2 3 4; do
        ck=checkpoints/collision/${m}_seed${s}.pt
        [[ -f "$ck" ]] && python eval.py --ckpt "$ck" --data-dir data/collision \
          --out results_prxq/collision/${m}_seed${s}.csv 2>/dev/null
      done; done
      for m in fidelityno mlp deepsets bidir; do
        python scripts/eval_recalibrated.py \
          --ckpts checkpoints/collision/${m}_seed{0,1,2,3,4}.pt \
          --data-root data/collision --calib-frac 0.1 --mode linear \
          --out results_prxq/collision/recalibrated_${m}.csv
      done
      # Diamond SDP baseline
      python scripts/eval_diamond_sdp.py --data-dir data/collision \
        --splits id_test,length_ood,family_ood --n-eval 1024 \
        --out results_prxq/collision/diamond_sdp_marginal.csv
      python scripts/eval_diamond_sdp.py --data-dir data/device_regime \
        --splits id_test,length_ood,family_ood --n-eval 1024 \
        --out results_prxq/real_hardware/diamond_sdp_device.csv
      # Two-qubit d=4 (uses v1 ckpts; data already exists)
      mkdir -p results_prxq/two_qubit_d4
      python scripts/eval_analytic.py \
        --data-dir data/benchmarks/two_qubit_order_sensitive \
        --out results_prxq/two_qubit_d4/analytic.csv
      python scripts/eval_mc.py \
        --data-dir data/benchmarks/two_qubit_order_sensitive \
        --out results_prxq/two_qubit_d4/mc.csv \
        --budgets 10,100,1000 --max-eval 1024
      for m in fidelityno mlp deepsets bidir gnn generic_gnn; do
        python scripts/eval_recalibrated.py \
          --ckpts checkpoints/order_sensitive_20260610_1638_fix/${m}_two_qubit_order_sensitive_seed{0,1,2,3,4}.pt \
          --data-root data/benchmarks/two_qubit_order_sensitive \
          --calib-frac 0.1 --mode linear \
          --out results_prxq/two_qubit_d4/recalibrated_${m}.csv
      done
      ;;
  esac
  case "$1" in
    figures|all)
      echo "=== [phase figures] building tables and plots ==="
      python scripts/build_central_figure.py
      python scripts/print_collision_table.py | tee \
        results_prxq/collision/POOLED_TABLE.txt
      python scripts/build_unified_paper_figure.py
      ;;
  esac
}

if [[ "${PHASE}" == "all" ]]; then
  for p in data train eval figures; do run_phase "$p"; done
else
  run_phase "${PHASE}"
fi

echo "=== reproduction complete ==="
echo "Tables:    results_prxq/collision/POOLED_TABLE.txt"
echo "Figure:    results_prxq/figures/central_sample_complexity.pdf"
echo "Full log:  RESULTS_PRXQ.md"
