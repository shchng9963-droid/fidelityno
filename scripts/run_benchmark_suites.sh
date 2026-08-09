#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python}"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
N_TRAIN=${N_TRAIN:-10000}
N_TEST=${N_TEST:-2000}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-128}
DEVICE=${DEVICE:-cuda}
MC_BUDGETS=${MC_BUDGETS:-10,100}
MC_MAX_EVAL=${MC_MAX_EVAL:-1000}
BENCH_ROOT=${BENCH_ROOT:-data/benchmarks}
OUT_ROOT=${OUT_ROOT:-results/benchmarks}
mkdir -p logs checkpoints "$OUT_ROOT"

echo "[$(date '+%F %T')] generate benchmark datasets: train=${N_TRAIN} test=${N_TEST} root=${BENCH_ROOT}"
"$PYTHON" scripts/gen_benchmarks.py --out-root "$BENCH_ROOT" --n-train "$N_TRAIN" --n-test "$N_TEST" --seed 0

for bench in single_qubit_mixed single_qubit_lindblad_holdout two_qubit_mixed two_qubit_order_sensitive; do
  data_dir="$BENCH_ROOT/$bench"
  out_dir="$OUT_ROOT/$bench"
  mkdir -p "$out_dir"
  echo "[$(date '+%F %T')] benchmark=$bench analytical baselines"
  "$PYTHON" scripts/eval_analytic.py --data-dir "$data_dir" --out "$out_dir/analytic.csv"
  echo "[$(date '+%F %T')] benchmark=$bench MC budgets=${MC_BUDGETS} max_eval=${MC_MAX_EVAL}"
  "$PYTHON" scripts/eval_mc.py --data-dir "$data_dir" --out "$out_dir/mc.csv" --budgets "$MC_BUDGETS" --max-eval "$MC_MAX_EVAL"
  for seed in 0 1 2 3 4; do
    for model in fidelityno gnn generic_gnn mlp; do
      ckpt="${model}_${bench}_seed${seed}.pt"
      echo "[$(date '+%F %T')] benchmark=$bench train model=$model seed=$seed"
      "$PYTHON" train.py \
        model.name=$model \
        seed=$seed \
        data.train="$data_dir/train.npz" \
        data.val="$data_dir/id_test.npz" \
        train.epochs="$EPOCHS" \
        train.batch_size="$BATCH" \
        train.ckpt_name="$ckpt" \
        device="$DEVICE"
      echo "[$(date '+%F %T')] benchmark=$bench eval model=$model seed=$seed"
      "$PYTHON" eval.py --ckpt "checkpoints/$ckpt" --data-dir "$data_dir" --out "$out_dir/${model}_seed${seed}.csv"
      if [[ "$model" == "fidelityno" || "$model" == "gnn" || "$model" == "generic_gnn" || "$model" == "mlp" ]]; then
        "$PYTHON" scripts/eval_calibrated.py --ckpt "checkpoints/$ckpt" --data-dir "$data_dir" --out "$out_dir/${model}_seed${seed}_calibrated.csv"
      fi
    done
  done
  "$PYTHON" - "$out_dir" <<'PY'
from pathlib import Path
import sys, pandas as pd
out=Path(sys.argv[1])
main=[out/'analytic.csv', out/'mc.csv'] + [f for f in sorted(out.glob('*_seed*.csv')) if 'calibrated' not in f.name]
pd.concat([pd.read_csv(f) for f in main if f.exists()], ignore_index=True).to_csv(out/'summary.csv', index=False)
cal=sorted(out.glob('*_calibrated.csv'))
if cal:
    pd.concat([pd.read_csv(f) for f in cal], ignore_index=True).to_csv(out/'calibrated_summary.csv', index=False)
print('wrote', out/'summary.csv')
PY
 done
"$PYTHON" scripts/build_order_sensitive_summary.py || true
"$PYTHON" - <<'PY'
from pathlib import Path
import pandas as pd
rows=[]
for summary in Path('results/benchmarks').glob('*/summary.csv'):
    bench=summary.parent.name
    df=pd.read_csv(summary); df.insert(0,'benchmark',bench); rows.append(df)
if rows:
    pd.concat(rows, ignore_index=True).to_csv('results/benchmarks/summary_all.csv', index=False)
cal_rows=[]
for summary in Path('results/benchmarks').glob('*/calibrated_summary.csv'):
    bench=summary.parent.name
    df=pd.read_csv(summary); df.insert(0,'benchmark',bench); cal_rows.append(df)
if cal_rows:
    pd.concat(cal_rows, ignore_index=True).to_csv('results/benchmarks/calibrated_summary_all.csv', index=False)
print('benchmark aggregation done')
PY
