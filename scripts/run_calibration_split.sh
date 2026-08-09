#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python}"
export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
DATA_DIR=${DATA_DIR:-data/calibration_split}
OUT_DIR=${OUT_DIR:-results/ablations/calibration_split}
CKPT_DIR=${CKPT_DIR:-checkpoints/ablations/calibration_split}
N_TRAIN=${N_TRAIN:-20000}
N_CALIB=${N_CALIB:-3000}
N_TEST=${N_TEST:-3000}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-128}
DEVICE=${DEVICE:-cuda}
SEEDS=${SEEDS:-0 1 2 3 4}
MODELS=${MODELS:-fidelityno gnn mlp}
HOLDOUT_FAMILY=${HOLDOUT_FAMILY:-pauli}
mkdir -p "$DATA_DIR" "$OUT_DIR" "$CKPT_DIR" logs
start_ts=$(date '+%Y%m%d_%H%M%S')
run_id="calibration_split_${start_ts}"
echo "[$(date '+%F %T')] run_id=${run_id} data=${DATA_DIR} out=${OUT_DIR} ckpt=${CKPT_DIR} n_train=${N_TRAIN} n_calib=${N_CALIB} n_test=${N_TEST} epochs=${EPOCHS} batch=${BATCH} device=${DEVICE} models=${MODELS} seeds=${SEEDS}"

if [[ "${FORCE_REGEN_DATA:-0}" == "1" || ! -f "$DATA_DIR/calib.npz" || ! -f "$DATA_DIR/train.npz" ]]; then
  "$PYTHON" scripts/gen_data.py \
    --outdir "$DATA_DIR" \
    --n-train "$N_TRAIN" \
    --n-calib "$N_CALIB" \
    --n-test "$N_TEST" \
    --seed 0 \
    --holdout-family "$HOLDOUT_FAMILY"
fi

"$PYTHON" - <<'CHECKPY'
from pathlib import Path
import numpy as np, os, json
root=Path(os.environ.get('DATA_DIR','data/calibration_split'))
for name in ['train','calib','id_test','length_ood','family_ood']:
    p=root/f'{name}.npz'
    if not p.exists(): raise SystemExit(f'missing required split: {p}')
    d=np.load(p)
    print(f'{p}: x={d["x"].shape} lengths={sorted(set(d["length"].tolist()))} y_mean={float(d["y"].mean()):.6f} y_std={float(d["y"].std()):.6f}')
print((root/'manifest.json').read_text()[:1000])
CHECKPY

rm -f "$OUT_DIR"/*_seed*.csv "$OUT_DIR"/*_seed*_calibrated.csv "$OUT_DIR"/summary.csv "$OUT_DIR"/calibrated_summary.csv "$OUT_DIR"/aggregate.csv "$OUT_DIR"/calibrated_aggregate.csv
for seed in $SEEDS; do
  for model in $MODELS; do
    ckpt="${model}_seed${seed}.pt"
    echo "[$(date '+%F %T')] train model=${model} seed=${seed}"
    "$PYTHON" train.py \
      model.name="$model" \
      model.head_type=quantile \
      seed="$seed" \
      data.train="$DATA_DIR/train.npz" \
      data.val="$DATA_DIR/calib.npz" \
      train.epochs="$EPOCHS" \
      train.batch_size="$BATCH" \
      train.ckpt_dir="$CKPT_DIR" \
      train.ckpt_name="$ckpt" \
      device="$DEVICE"
    echo "[$(date '+%F %T')] eval raw model=${model} seed=${seed}"
    "$PYTHON" eval.py --ckpt "$CKPT_DIR/$ckpt" --data-dir "$DATA_DIR" --out "$OUT_DIR/${model}_seed${seed}.csv"
    echo "[$(date '+%F %T')] eval calibrated model=${model} seed=${seed}"
    "$PYTHON" scripts/eval_calibrated.py --ckpt "$CKPT_DIR/$ckpt" --data-dir "$DATA_DIR" --out "$OUT_DIR/${model}_seed${seed}_calibrated.csv"
  done
done

"$PYTHON" - <<'AGGPY'
from pathlib import Path
import pandas as pd, os
out=Path(os.environ.get('OUT_DIR','results/ablations/calibration_split'))
raw_files=sorted(p for p in out.glob('*_seed*.csv') if not p.name.endswith('_calibrated.csv'))
cal_files=sorted(out.glob('*_seed*_calibrated.csv'))
if not raw_files: raise SystemExit('no raw csv files found')
raw=pd.concat([pd.read_csv(f) for f in raw_files], ignore_index=True)
cal=pd.concat([pd.read_csv(f) for f in cal_files], ignore_index=True) if cal_files else pd.DataFrame()
raw.to_csv(out/'summary.csv', index=False)
if not cal.empty: cal.to_csv(out/'calibrated_summary.csv', index=False)
for df, name in [(raw,'aggregate.csv'), (cal,'calibrated_aggregate.csv')]:
    if df.empty: continue
    agg=(df.groupby(['model','split'], as_index=False)
           .agg(mae_mean=('mae','mean'), mae_std=('mae','std'),
                pinball_mean=('pinball','mean'), crps_mean=('crps','mean'),
                ece_mean=('ece','mean'), ece_std=('ece','std'),
                latency_ms_mean=('latency_ms','mean')))
    agg.to_csv(out/name, index=False)
    print('\n'+name)
    print(agg.to_string(index=False))
print('wrote', out/'summary.csv', 'rows', len(raw))
if not cal.empty: print('wrote', out/'calibrated_summary.csv', 'rows', len(cal))
AGGPY

echo "[$(date '+%F %T')] done run_id=${run_id}"
