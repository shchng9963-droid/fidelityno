#!/usr/bin/env bash
# Evaluate everything on the collision dataset and produce a unified
# pooled MAE table (PRXQ P1.1).

set -e -u -o pipefail
cd "$(dirname "$0")/.."

DATA_DIR="data/collision"
CKPT_DIR="checkpoints/collision"
OUT_ROOT="results_prxq/collision"
mkdir -p "$OUT_ROOT"

if [[ ! -d "$CKPT_DIR" ]] || [[ -z "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]]; then
  echo "[error] no checkpoints in $CKPT_DIR; run scripts/train_collision.sh first."
  exit 1
fi

# 1) Analytic baselines (product_bound / fvg / diamond / analytic_best)
python scripts/eval_analytic.py --data-dir "$DATA_DIR" --out "$OUT_ROOT/analytic.csv"

# 2) Monte Carlo at K in {10, 100, 1000}
python scripts/eval_mc.py --data-dir "$DATA_DIR" --out "$OUT_ROOT/mc.csv" \
  --budgets 10,100,1000 --max-eval 1024

# 3) DFE at multiple budgets on the *id_test* and *length_ood* and
#    *family_ood* splits.  Each split contributes one CSV row per S.
for split in id_test length_ood family_ood; do
  python scripts/eval_dfe.py \
    --data "$DATA_DIR/${split}.npz" \
    --pauli-budgets 10,30,100,300,1000 \
    --M 200 --n-eval 1024 \
    --out "$OUT_ROOT/dfe_${split}.csv"
done

# 4) NN models on all splits using existing eval.py
ckpts=()
for f in "$CKPT_DIR"/*.pt; do
  ckpts+=("$f")
done
for ckpt in "${ckpts[@]}"; do
  name="$(basename "$ckpt" .pt)"
  python eval.py --ckpt "$ckpt" --data-dir "$DATA_DIR" \
    --out "$OUT_ROOT/${name}.csv" --device cpu
done

# 5) Aggregate
python scripts/build_collision_summary.py
