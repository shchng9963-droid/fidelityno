#!/usr/bin/env bash
# Resume representation ablation — only run missing experiments
set -euo pipefail
cd /home/wangshuchang/fidelityno
ENV=/home/wangshuchang/miniforge3/envs/fidelityno/bin
export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
BASE_DATA_DIR=data/ablations/representation
OUT_DIR=results/ablations/representation
CKPT_DIR=checkpoints/ablations/representation
N_TRAIN=20000
N_TEST=3000
EPOCHS=30
BATCH=128
DEVICE=cuda
HOLDOUT_FAMILY=pauli
mkdir -p "$BASE_DATA_DIR" "$OUT_DIR" "$CKPT_DIR" logs

echo "[$(date '+%F %T')] Resuming representation ablation — filling missing runs"

# Generate PTM data if missing
ptm_dir="$BASE_DATA_DIR/ptm"
mkdir -p "$ptm_dir"
if [[ ! -f "$ptm_dir/train.npz" || ! -f "$ptm_dir/manifest.json" ]]; then
  echo "[$(date '+%F %T')] generate representation=ptm"
  "$ENV/python" scripts/gen_data.py \
    --outdir "$ptm_dir" \
    --n-train "$N_TRAIN" \
    --n-test "$N_TEST" \
    --seed 0 \
    --holdout-family "$HOLDOUT_FAMILY" \
    --representation ptm
fi

# List of missing experiments
declare -a MISSING=(
  "gnn raw_choi_flat 4"
  "fidelityno ptm 0"
  "fidelityno ptm 1"
  "fidelityno ptm 2"
  "fidelityno ptm 3"
  "fidelityno ptm 4"
  "gnn ptm 0"
  "gnn ptm 1"
  "gnn ptm 2"
  "gnn ptm 3"
  "gnn ptm 4"
)

for entry in "${MISSING[@]}"; do
  read -r model rep seed <<< "$entry"
  data_dir="$BASE_DATA_DIR/$rep"
  ckpt="${model}_${rep}_seed${seed}.pt"
  raw_csv="$OUT_DIR/${model}_${rep}_seed${seed}.csv"

  if [[ -f "$raw_csv" ]]; then
    echo "[$(date '+%F %T')] SKIP (exists): model=$model rep=$rep seed=$seed"
    continue
  fi

  echo "[$(date '+%F %T')] train model=$model rep=$rep seed=$seed"
  "$ENV/python" train.py \
    model.name="$model" \
    model.head_type=quantile \
    seed="$seed" \
    data.train="$data_dir/train.npz" \
    data.val="$data_dir/id_test.npz" \
    train.epochs="$EPOCHS" \
    train.batch_size="$BATCH" \
    train.ckpt_dir="$CKPT_DIR" \
    train.ckpt_name="$ckpt" \
    device="$DEVICE"

  echo "[$(date '+%F %T')] eval model=$model rep=$rep seed=$seed"
  "$ENV/python" eval.py --ckpt "$CKPT_DIR/$ckpt" --data-dir "$data_dir" --out "$raw_csv"

  # Tag with representation column
  "$ENV/python" -c "
import pandas as pd
df=pd.read_csv('$raw_csv')
df['representation']='$rep'
df.to_csv('$raw_csv', index=False)
"
done

echo "[$(date '+%F %T')] All missing runs complete. Aggregating..."

# Aggregate all results
"$ENV/python" - <<'AGGPY'
from pathlib import Path
import pandas as pd
out = Path("results/ablations/representation")
files = sorted(out.glob("*_seed*.csv"))
if not files:
    raise SystemExit("No result CSVs found")
dfs = [pd.read_csv(f) for f in files]
combined = pd.concat(dfs, ignore_index=True)
combined.to_csv(out / "summary.csv", index=False)
print(f"Combined {len(files)} CSVs -> summary.csv ({len(combined)} rows)")

# Aggregate: mean ± std over seeds, grouped by model, representation, split
agg = combined.groupby(["model", "representation", "split"]).agg(
    mae_mean=("mae", "mean"), mae_std=("mae", "std"),
    pinball_mean=("pinball", "mean"), pinball_std=("pinball", "std"),
    crps_mean=("crps", "mean"), crps_std=("crps", "std"),
    ece_mean=("ece", "mean"), ece_std=("ece", "std"),
    latency_mean=("latency_ms", "mean"),
).reset_index()
agg.to_csv(out / "aggregate.csv", index=False)
print(f"Aggregate -> aggregate.csv ({len(agg)} rows)")
print(agg.to_string())
AGGPY

echo "[$(date '+%F %T')] DONE representation ablation resume"
