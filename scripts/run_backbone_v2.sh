#!/usr/bin/env bash
# C5 follow-up: complete 5 seeds for original gru_causal + gru_bidir, AND run
# the param-matched comparisons that the original C5 framing missed.
#
# Why this exists: the original C5 ran gru_bidir at d=128 (432k params) and
# compared it to FidelityNO-G at d=128/layers=4 (202k params). That is a
# 2x param disadvantage for fnoG, so the win is confounded.
#
# This script adds:
#   1. seeds 3, 4 for gru_causal & gru_bidir at the original d=128 config
#      (so the original C5 cell has 5 seeds for paper-grade error bars).
#   2. A param-MATCHED low pair: gru_bidir d=80 (~203k) vs fnoG d=128 layers=4
#      (~202k). The fnoG side already exists as the production run, so we only
#      need the gru side.
#   3. A param-MATCHED high pair: gru_bidir d=128 (~432k) vs fnoG d=192/layers=4
#      (~451k). The gru side will exist after step 1; we need fnoG d=192.
#
# All run on the production single-qubit-mixed 200k dataset (data/), 30 epochs,
# batch 128, head=quantile, aux=true.

set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}
cd /home/wangshuchang/fidelityno
ENV=/home/wangshuchang/miniforge3/envs/fidelityno/bin
export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
DATA_DIR=${DATA_DIR:-data}
OUT_DIR=${OUT_DIR:-results/ablations/backbone_v2}
CKPT_DIR=${CKPT_DIR:-checkpoints/ablations/backbone_v2}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-128}
DEVICE=${DEVICE:-cuda}
mkdir -p "$OUT_DIR" "$CKPT_DIR" logs

# Configurations to compare:
# label | model.name | d_model | depth | layers | bidir(only used for gru) | seeds
# Format: label:model:d:depth:layers:bidir:seeds
RUNS=(
  # T1: complete original C5 to 5 seeds (re-runs 0-4 cleanly).
  "gru_causal_d128:fidelityno_gru:128:4:4:false:0 1 2 3 4"
  "gru_bidir_d128:fidelityno_gru:128:4:4:true:0 1 2 3 4"
  # T2 LOW: param-matched gru_bidir at fnoG's 200k budget.
  "gru_bidir_d80:fidelityno_gru:80:4:4:true:0 1 2 3 4"
  # T2 HIGH: scale up fnoG to gru_bidir_d128's 432k budget.
  "fnoG_d192_l4:gnn:192:4:4:false:0 1 2 3 4"
)

start_ts=$(date '+%Y%m%d_%H%M%S')
run_id="backbone_v2_${start_ts}"
echo "[$(date '+%F %T')] run_id=${run_id} GPU=${CUDA_VISIBLE_DEVICES} epochs=${EPOCHS}"

for spec in "${RUNS[@]}"; do
  IFS=':' read -r label mname d depth layers bidir seeds <<<"$spec"
  for seed in $seeds; do
    ckpt="${label}_seed${seed}.pt"
    csv="$OUT_DIR/${label}_seed${seed}.csv"
    if [[ -s "$csv" && "${FORCE_REDO:-0}" != "1" ]]; then
      echo "[$(date '+%F %T')] skip (csv exists): $csv"
      continue
    fi
    echo "[$(date '+%F %T')] train ${label} seed=${seed} (model=${mname} d=${d} depth=${depth} layers=${layers} bidir=${bidir})"
    if [[ "$mname" == "fidelityno_gru" ]]; then
      $ENV/python train.py \
        model.name="$mname" \
        model.d_model="$d" \
        model.depth="$depth" \
        +model.bidir="$bidir" \
        model.head_type=quantile \
        model.aux=true \
        seed="$seed" \
        data.train="$DATA_DIR/train.npz" \
        data.val="$DATA_DIR/id_test.npz" \
        train.epochs="$EPOCHS" \
        train.batch_size="$BATCH" \
        train.ckpt_dir="$CKPT_DIR" \
        train.ckpt_name="$ckpt" \
        device="$DEVICE"
    else
      # fnoG (gnn) -- uses layers, not depth
      $ENV/python train.py \
        model.name="$mname" \
        model.d_model="$d" \
        model.layers="$layers" \
        model.head_type=quantile \
        model.aux=true \
        seed="$seed" \
        data.train="$DATA_DIR/train.npz" \
        data.val="$DATA_DIR/id_test.npz" \
        train.epochs="$EPOCHS" \
        train.batch_size="$BATCH" \
        train.ckpt_dir="$CKPT_DIR" \
        train.ckpt_name="$ckpt" \
        device="$DEVICE"
    fi
    echo "[$(date '+%F %T')] eval ${label} seed=${seed}"
    $ENV/python eval.py --ckpt "$CKPT_DIR/$ckpt" --data-dir "$DATA_DIR" --out "$csv"
    # Patch the model column in the CSV to use our `label` (not cfg.model.name)
    # This is the fix for the prior labeling bug.
    $ENV/python - "$csv" "$label" <<'PYFIX'
import sys, pandas as pd
csv, label = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv)
df['model'] = label
df.to_csv(csv, index=False)
PYFIX
  done
done

# Aggregate using filename-derived label (proper, not cfg.model.name).
$ENV/python - <<'AGGPY'
from pathlib import Path
import os, re, pandas as pd
out = Path(os.environ.get('OUT_DIR','results/ablations/backbone_v2'))
files = sorted(out.glob('*_seed*.csv'))
if not files:
    raise SystemExit('no csv files found')
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(out/'summary.csv', index=False)
agg = (df.groupby(['model','split'], as_index=False)
         .agg(mae_mean=('mae','mean'), mae_std=('mae','std'),
              pinball_mean=('pinball','mean'), crps_mean=('crps','mean'),
              ece_mean=('ece','mean'), ece_std=('ece','std'),
              latency_ms_mean=('latency_ms','mean'),
              n_seeds=('seed','nunique')))
agg.to_csv(out/'aggregate.csv', index=False)
print(agg.to_string(index=False))

# Also produce a length-resolved aggregate for length-OOD analysis
agg_len = (df.groupby(['model','split','length'], as_index=False)
             .agg(mae_mean=('mae','mean'), mae_std=('mae','std'),
                  n_seeds=('seed','nunique')))
agg_len.to_csv(out/'aggregate_by_length.csv', index=False)
AGGPY
echo "[$(date '+%F %T')] done run_id=${run_id}"
