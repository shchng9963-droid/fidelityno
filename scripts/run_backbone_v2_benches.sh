#!/usr/bin/env bash
# C5 generalization: gru_bidir on the 2-qubit and Lindblad benchmark suites.
# 5 seeds each, paired with FidelityNO-G runs already in
# data/benchmarks/{two_qubit_mixed,single_qubit_lindblad_holdout}.
#
# Datasets are 10k train (matches the production run_benchmark_suites.sh).
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python}"
export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-128}
DEVICE=${DEVICE:-cuda}
SEEDS=${SEEDS:-0 1 2 3 4}
BENCHES=${BENCHES:-two_qubit_mixed single_qubit_lindblad_holdout}
mkdir -p logs

start_ts=$(date '+%Y%m%d_%H%M%S')
run_id="backbone_v2_benches_${start_ts}"
echo "[$(date '+%F %T')] run_id=${run_id} GPU=${CUDA_VISIBLE_DEVICES} benches=${BENCHES} seeds=${SEEDS}"

for bench in $BENCHES; do
  data_dir="data/benchmarks/$bench"
  out_dir="results/benchmarks/$bench"
  ckpt_dir="checkpoints/ablations/backbone_v2_${bench}"
  mkdir -p "$out_dir" "$ckpt_dir"
  for seed in $SEEDS; do
    label="gru_bidir_d128"
    ckpt="${label}_seed${seed}.pt"
    csv="$out_dir/${label}_seed${seed}.csv"
    if [[ -s "$csv" && "${FORCE_REDO:-0}" != "1" ]]; then
      echo "[$(date '+%F %T')] skip $csv"
      continue
    fi
    echo "[$(date '+%F %T')] train bench=${bench} ${label} seed=${seed}"
    "$PYTHON" train.py \
      model.name=fidelityno_gru \
      model.d_model=128 \
      model.depth=4 \
      +model.bidir=true \
      model.head_type=quantile \
      model.aux=true \
      seed="$seed" \
      data.train="$data_dir/train.npz" \
      data.val="$data_dir/id_test.npz" \
      train.epochs="$EPOCHS" \
      train.batch_size="$BATCH" \
      train.ckpt_dir="$ckpt_dir" \
      train.ckpt_name="$ckpt" \
      device="$DEVICE"
    echo "[$(date '+%F %T')] eval bench=${bench} ${label} seed=${seed}"
    "$PYTHON" eval.py --ckpt "$ckpt_dir/$ckpt" --data-dir "$data_dir" --out "$csv"
    # Re-label so it doesn't collide with cfg.model.name='fidelityno_gru'
    "$PYTHON" - "$csv" "$label" <<'PYFIX'
import sys, pandas as pd
df=pd.read_csv(sys.argv[1]); df['model']=sys.argv[2]; df.to_csv(sys.argv[1], index=False)
PYFIX
  done
  # Aggregate this benchmark (gru_bidir + existing baselines + analytic + mc).
  "$PYTHON" - "$out_dir" <<'AGGPY'
import sys, pandas as pd
from pathlib import Path
out = Path(sys.argv[1])
files = sorted(p for p in out.glob('*.csv')
               if p.name not in ('summary.csv','calibrated_summary.csv','aggregate.csv','calibrated_aggregate.csv','length_aggregate.csv'))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(out/'summary.csv', index=False)
agg = (df.groupby(['model','split'], as_index=False)
         .agg(mae_mean=('mae','mean'), mae_std=('mae','std'),
              pinball_mean=('pinball','mean'), crps_mean=('crps','mean'),
              ece_mean=('ece','mean'),
              latency_ms_mean=('latency_ms','mean'),
              n_seeds=('seed','nunique')))
agg.to_csv(out/'aggregate.csv', index=False)
print(agg.to_string(index=False))
AGGPY
done
echo "[$(date '+%F %T')] done run_id=${run_id}"
