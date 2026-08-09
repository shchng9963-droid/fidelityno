#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python}"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p logs results checkpoints
for seed in 0 1 2 3 4; do
  echo "[$(date '+%F %T')] train FidelityNO-T no-aux seed=${seed}"
  "$PYTHON" train.py \
    model.name=fidelityno_noaux \
    model.aux=false \
    seed=${seed} \
    train.epochs=${EPOCHS:-30} \
    train.batch_size=${BATCH:-128} \
    train.ckpt_name=fidelityno_noaux_seed${seed}.pt \
    device=${DEVICE:-cuda}
  echo "[$(date '+%F %T')] eval FidelityNO-T no-aux seed=${seed}"
  "$PYTHON" eval.py --ckpt checkpoints/fidelityno_noaux_seed${seed}.pt --out results/fidelityno_noaux_seed${seed}.csv
  "$PYTHON" scripts/eval_calibrated.py --ckpt checkpoints/fidelityno_noaux_seed${seed}.pt --out results/fidelityno_noaux_seed${seed}_calibrated.csv
 done
"$PYTHON" - <<'PY'
from pathlib import Path
import pandas as pd
files=[Path('results/analytic.csv'), Path('results/mc.csv')] + sorted(Path('results').glob('*_seed*.csv'))
# exclude calibrated from main uncalibrated aggregate
files=[f for f in files if f.exists() and 'calibrated' not in f.name]
pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv('results/summary.csv', index=False)
cal_files=sorted(Path('results').glob('*_calibrated.csv'))
if cal_files:
    pd.concat([pd.read_csv(f) for f in cal_files], ignore_index=True).to_csv('results/calibrated_summary.csv', index=False)
print('updated summary and calibrated_summary')
PY
"$PYTHON" scripts/make_figures.py
