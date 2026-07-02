#!/usr/bin/env bash
# DeepSets width sweep — verifies the MAE plateau at d=768 is a capacity/expressivity
# floor rather than undertraining. Reuses the choi_hermitian dataset from the
# representation ablation (20k train / 3k test, 30 epochs, mixed single-qubit benchmark).
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}
cd /home/wangshuchang/fidelityno
ENV=/home/wangshuchang/miniforge3/envs/fidelityno/bin
export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

DATA_DIR=${DATA_DIR:-data/ablations/representation/choi_hermitian}
OUT_DIR=${OUT_DIR:-results/ablations/deepsets_width}
CKPT_DIR=${CKPT_DIR:-checkpoints/ablations/deepsets_width}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-128}
SEEDS=${SEEDS:-0 1 2}
WIDTHS=${WIDTHS:-128 256 512 768 1536}

mkdir -p "$OUT_DIR" "$CKPT_DIR" logs

for f in train.npz id_test.npz length_ood.npz family_ood.npz; do
  if [ ! -f "$DATA_DIR/$f" ]; then
    echo "[ERROR] missing $DATA_DIR/$f — run scripts/run_representation_ablation.sh first" >&2
    exit 1
  fi
done

for W in $WIDTHS; do
  for S in $SEEDS; do
    TAG=deepsets_w${W}_s${S}
    CKPT_FILE="${TAG}.pt"
    RAW_CSV="$OUT_DIR/${TAG}.csv"
    LOG="logs/${TAG}.log"
    if [ -f "$CKPT_DIR/$CKPT_FILE" ] && [ -f "$RAW_CSV" ]; then
      echo "[skip] $TAG (have ckpt + csv)"
      continue
    fi
    echo "[$(date '+%F %T')] train deepsets width=$W seed=$S"
    "$ENV/python" train.py \
        model=deepsets \
        model.d_model="$W" \
        seed="$S" \
        data.train="$DATA_DIR/train.npz" \
        data.val="$DATA_DIR/id_test.npz" \
        train.epochs="$EPOCHS" \
        train.batch_size="$BATCH" \
        train.ckpt_dir="$CKPT_DIR" \
        train.ckpt_name="$CKPT_FILE" \
        device=cuda >"$LOG" 2>&1 || { echo "[FAIL] train $TAG (see $LOG)"; continue; }
    "$ENV/python" eval.py --ckpt "$CKPT_DIR/$CKPT_FILE" --data-dir "$DATA_DIR" --out "$RAW_CSV" >>"$LOG" 2>&1
    "$ENV/python" - <<PY >>"$LOG"
import pandas as pd
df=pd.read_csv("$RAW_CSV"); df['width']=$W; df.to_csv("$RAW_CSV",index=False)
PY
  done
done

"$ENV/python" - <<'AGG'
from pathlib import Path
import os, pandas as pd
out=Path(os.environ.get('OUT_DIR','results/ablations/deepsets_width'))
files=sorted(out.glob('deepsets_w*_s*.csv'))
if not files: raise SystemExit('no width-sweep csv files found')
df=pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(out/'summary.csv', index=False)
agg=(df.groupby(['width','split'], as_index=False)
       .agg(mae_mean=('mae','mean'), mae_std=('mae','std'),
            pinball_mean=('pinball','mean'),
            ece_mean=('ece','mean')))
agg.to_csv(out/'aggregate.csv', index=False)
print(agg.to_string(index=False))
AGG
echo "[$(date '+%F %T')] deepsets width sweep done"
