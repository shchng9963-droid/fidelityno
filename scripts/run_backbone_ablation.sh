#!/usr/bin/env bash
# C5: backbone ablation -- swap the causal-Transformer backbone for a stacked
# GRU (and bidirectional GRU) inside the same FidelityNO skeleton.
# Goal: defend the architecture choice in the paper. The "S4/Mamba" wording
# from the original brief is replaced with this concrete non-attention
# sequential ablation, since attention vs recurrence is the discriminating
# question (we already have causal-vs-bidir transformer via B6).
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python}"
export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
DATA_DIR=${DATA_DIR:-data}
OUT_DIR=${OUT_DIR:-results/ablations/backbone}
CKPT_DIR=${CKPT_DIR:-checkpoints/ablations/backbone}
export DATA_DIR OUT_DIR CKPT_DIR
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-128}
DEVICE=${DEVICE:-cuda}
SEEDS=${SEEDS:-0 1 2}
mkdir -p "$OUT_DIR" "$CKPT_DIR" logs
start_ts=$(date '+%Y%m%d_%H%M%S')
run_id="backbone_ablation_${start_ts}"
echo "[$(date '+%F %T')] run_id=${run_id} data=${DATA_DIR} out=${OUT_DIR} ckpt=${CKPT_DIR} epochs=${EPOCHS} seeds=${SEEDS}"

# Configurations to compare: model_name | depth | bidir | label
CONFIGS=(
  "fidelityno_gru:4:false:gru_causal"
  "fidelityno_gru:4:true:gru_bidir"
)

rm -f "$OUT_DIR"/*_seed*.csv "$OUT_DIR"/summary.csv "$OUT_DIR"/aggregate.csv
for seed in $SEEDS; do
  for cfg in "${CONFIGS[@]}"; do
    IFS=':' read -r mname depth bidir label <<<"$cfg"
    ckpt="${label}_seed${seed}.pt"
    echo "[$(date '+%F %T')] train ${label} seed=${seed} (model=${mname} depth=${depth} bidir=${bidir})"
    "$PYTHON" train.py \
      model.name="$mname" \
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
    echo "[$(date '+%F %T')] eval ${label} seed=${seed}"
    "$PYTHON" eval.py --ckpt "$CKPT_DIR/$ckpt" --data-dir "$DATA_DIR" --out "$OUT_DIR/${label}_seed${seed}.csv"
  done
done

"$PYTHON" - <<'AGGPY'
from pathlib import Path
import pandas as pd, os
out=Path(os.environ.get('OUT_DIR','results/ablations/backbone'))
files=sorted(out.glob('*_seed*.csv'))
if not files: raise SystemExit('no backbone ablation csv files found')
df=pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
# Tag with the variant label from the filename
df['variant']=df.apply(lambda r: r.get('model','?'), axis=1)
df.to_csv(out/'summary.csv', index=False)
agg=(df.groupby(['model','split'], as_index=False)
       .agg(mae_mean=('mae','mean'), mae_std=('mae','std'),
            pinball_mean=('pinball','mean'), crps_mean=('crps','mean'),
            ece_mean=('ece','mean'), latency_ms_mean=('latency_ms','mean')))
agg.to_csv(out/'aggregate.csv', index=False)
print(agg.to_string(index=False))
AGGPY
echo "[$(date '+%F %T')] done run_id=${run_id}"
