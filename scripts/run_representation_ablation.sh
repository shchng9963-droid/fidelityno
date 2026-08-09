#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python}"
export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
BASE_DATA_DIR=${BASE_DATA_DIR:-data/ablations/representation}
OUT_DIR=${OUT_DIR:-results/ablations/representation}
CKPT_DIR=${CKPT_DIR:-checkpoints/ablations/representation}
N_TRAIN=${N_TRAIN:-20000}
N_TEST=${N_TEST:-3000}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-128}
DEVICE=${DEVICE:-cuda}
SEEDS=${SEEDS:-0 1 2 3 4}
MODELS=${MODELS:-fidelityno gnn}
REPRESENTATIONS=${REPRESENTATIONS:-choi_hermitian raw_choi_flat compressed_hermitian ptm}
HOLDOUT_FAMILY=${HOLDOUT_FAMILY:-pauli}
mkdir -p "$BASE_DATA_DIR" "$OUT_DIR" "$CKPT_DIR" logs
start_ts=$(date '+%Y%m%d_%H%M%S')
run_id="representation_ablation_${start_ts}"
echo "[$(date '+%F %T')] run_id=${run_id} base_data=${BASE_DATA_DIR} out=${OUT_DIR} ckpt=${CKPT_DIR} n_train=${N_TRAIN} n_test=${N_TEST} epochs=${EPOCHS} batch=${BATCH} device=${DEVICE} reps=${REPRESENTATIONS} models=${MODELS} seeds=${SEEDS}"

rm -f "$OUT_DIR"/*_seed*.csv "$OUT_DIR"/summary.csv "$OUT_DIR"/aggregate.csv
for rep in $REPRESENTATIONS; do
  data_dir="$BASE_DATA_DIR/$rep"
  mkdir -p "$data_dir"
  if [[ "${FORCE_REGEN_DATA:-0}" == "1" || ! -f "$data_dir/train.npz" || ! -f "$data_dir/manifest.json" ]]; then
    echo "[$(date '+%F %T')] generate representation=${rep}"
    "$PYTHON" scripts/gen_data.py \
      --outdir "$data_dir" \
      --n-train "$N_TRAIN" \
      --n-test "$N_TEST" \
      --seed 0 \
      --holdout-family "$HOLDOUT_FAMILY" \
      --representation "$rep"
  fi
  CURRENT_REP="$rep" CURRENT_DATA_DIR="$data_dir" "$PYTHON" - <<'CHECKPY'
from pathlib import Path
import numpy as np, os
rep=os.environ['CURRENT_REP']; root=Path(os.environ['CURRENT_DATA_DIR'])
for name in ['train','id_test','length_ood','family_ood']:
    p=root/f'{name}.npz'
    if not p.exists(): raise SystemExit(f'missing {p}')
    d=np.load(p)
    print(f'rep={rep} {name}: x={d["x"].shape} y_mean={float(d["y"].mean()):.6f} y_std={float(d["y"].std()):.6f}')
CHECKPY
  for seed in $SEEDS; do
    for model in $MODELS; do
      ckpt="${model}_${rep}_seed${seed}.pt"
      raw_csv="$OUT_DIR/${model}_${rep}_seed${seed}.csv"
      echo "[$(date '+%F %T')] train model=${model} rep=${rep} seed=${seed}"
      "$PYTHON" train.py \
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
      echo "[$(date '+%F %T')] eval model=${model} rep=${rep} seed=${seed}"
      "$PYTHON" eval.py --ckpt "$CKPT_DIR/$ckpt" --data-dir "$data_dir" --out "$raw_csv"
      CURRENT_REP="$rep" CURRENT_CSV="$raw_csv" "$PYTHON" - <<'TAGPY'
import os, pandas as pd
p=os.environ['CURRENT_CSV']; rep=os.environ['CURRENT_REP']
df=pd.read_csv(p); df['representation']=rep; df.to_csv(p,index=False)
TAGPY
    done
  done
done

"$PYTHON" - <<'AGGPY'
from pathlib import Path
import pandas as pd, os
out=Path(os.environ.get('OUT_DIR','results/ablations/representation'))
files=sorted(out.glob('*_seed*.csv'))
if not files: raise SystemExit('no representation csv files found')
df=pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(out/'summary.csv', index=False)
agg=(df.groupby(['model','representation','split'], as_index=False)
       .agg(mae_mean=('mae','mean'), mae_std=('mae','std'),
            pinball_mean=('pinball','mean'), crps_mean=('crps','mean'),
            ece_mean=('ece','mean'), ece_std=('ece','std'),
            latency_ms_mean=('latency_ms','mean')))
agg.to_csv(out/'aggregate.csv', index=False)
print('wrote', out/'summary.csv', 'rows', len(df))
print('wrote', out/'aggregate.csv')
print(agg.to_string(index=False))
AGGPY

echo "[$(date '+%F %T')] done run_id=${run_id}"
