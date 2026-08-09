#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python}"
MAX_EVAL=${MC1000_MAX_EVAL:-500}
mkdir -p logs
for bench in single_qubit_mixed single_qubit_lindblad_holdout two_qubit_mixed; do
  echo "[$(date '+%F %T')] MC-1000 capped benchmark=$bench max_eval=$MAX_EVAL"
  "$PYTHON" scripts/eval_mc.py \
    --data-dir "data/benchmarks/$bench" \
    --out "results/benchmarks/$bench/mc1000_cap${MAX_EVAL}.csv" \
    --budgets 1000 \
    --max-eval "$MAX_EVAL"
done
"$PYTHON" - <<'PY'
from pathlib import Path
import pandas as pd
rows=[]
for f in Path('results/benchmarks').glob('*/mc1000_cap*.csv'):
    df=pd.read_csv(f)
    df.insert(0,'benchmark',f.parent.name)
    rows.append(df)
if rows:
    out=Path('results/benchmarks/mc1000_capped_summary.csv')
    pd.concat(rows, ignore_index=True).to_csv(out, index=False)
    print('wrote', out)
PY
