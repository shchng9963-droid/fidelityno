#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python}"
export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
DATA_DIR=${DATA_DIR:-data}
OUT_DIR=${OUT_DIR:-results/ablations/head}
CKPT_DIR=${CKPT_DIR:-checkpoints/ablations/head}
export DATA_DIR OUT_DIR CKPT_DIR
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-128}
DEVICE=${DEVICE:-cuda}
SEEDS=${SEEDS:-0 1 2 3 4}
HEADS=${HEADS:-quantile gaussian scalar}
AUX=${AUX:-true}
mkdir -p "$OUT_DIR" "$CKPT_DIR" logs
start_ts=$(date '+%Y%m%d_%H%M%S')
run_id="head_ablation_${start_ts}"
echo "[$(date '+%F %T')] run_id=${run_id} data=${DATA_DIR} out=${OUT_DIR} ckpt=${CKPT_DIR} epochs=${EPOCHS} batch=${BATCH} device=${DEVICE} heads=${HEADS} seeds=${SEEDS} aux=${AUX}"
"$PYTHON" - <<'CHECKPY'
from pathlib import Path
import numpy as np, os
root=Path(os.environ.get('DATA_DIR','data'))
for name in ['train','id_test','length_ood','family_ood']:
    p=root/f'{name}.npz'
    if not p.exists(): raise SystemExit(f'missing required split: {p}')
    d=np.load(p)
    print(f'{p}: x={d["x"].shape} y_mean={float(d["y"].mean()):.6f} y_std={float(d["y"].std()):.6f}')
CHECKPY
rm -f "$OUT_DIR"/fidelityno_*_seed*.csv "$OUT_DIR"/summary.csv "$OUT_DIR"/aggregate.csv
for seed in $SEEDS; do
  for head in $HEADS; do
    ckpt="fidelityno_${head}_seed${seed}.pt"
    echo "[$(date '+%F %T')] train head=${head} seed=${seed}"
    "$PYTHON" train.py \
      model.name=fidelityno \
      model.head_type="$head" \
      model.aux="$AUX" \
      seed="$seed" \
      data.train="$DATA_DIR/train.npz" \
      data.val="$DATA_DIR/id_test.npz" \
      train.epochs="$EPOCHS" \
      train.batch_size="$BATCH" \
      train.ckpt_dir="$CKPT_DIR" \
      train.ckpt_name="$ckpt" \
      device="$DEVICE"
    echo "[$(date '+%F %T')] eval head=${head} seed=${seed}"
    "$PYTHON" eval.py --ckpt "$CKPT_DIR/$ckpt" --data-dir "$DATA_DIR" --out "$OUT_DIR/fidelityno_${head}_seed${seed}.csv"
  done
done
"$PYTHON" - <<'AGGPY'
from pathlib import Path
import pandas as pd, os
out=Path(os.environ.get('OUT_DIR','results/ablations/head'))
files=sorted(out.glob('fidelityno_*_seed*.csv'))
if not files: raise SystemExit('no head ablation csv files found')
df=pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(out/'summary.csv', index=False)
agg=(df.groupby(['model','head_type','split'], as_index=False)
       .agg(mae_mean=('mae','mean'), mae_std=('mae','std'),
            pinball_mean=('pinball','mean'), crps_mean=('crps','mean'),
            ece_mean=('ece','mean'), latency_ms_mean=('latency_ms','mean')))
agg.to_csv(out/'aggregate.csv', index=False)
print('wrote', out/'summary.csv', 'rows', len(df))
print('wrote', out/'aggregate.csv')
print(agg.to_string(index=False))
AGGPY
echo "[$(date '+%F %T')] done run_id=${run_id}"
