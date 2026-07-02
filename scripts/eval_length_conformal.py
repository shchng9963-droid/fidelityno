#!/usr/bin/env python
"""Length-conditioned split-conformal calibration (C1).

Strategy
--------
The default eval_calibrated.py fits a single per-quantile offset on
the ID calibration split (lengths n in {2,4,8,16}) and applies it
uniformly to length-OOD lengths {24,32,48}. Empirically that is too
conservative under length shift (over-coverage ~1).

Length-conditioned conformal:
1. Split id_test (the calibration source) into per-length buckets.
2. For each nominal level a and each length L_train, fit
   o_a(L) = Quantile_a(y - q_a(x)) on the bucket of length L.
3. Fit a 1-D regression o_a(L) ~ alpha_a + beta_a * f(L) where
   f(L) = log(L) (default; we also try f(L)=L) using the
   four train lengths.
4. At test time, for a sequence of length L, apply offset
   o_a_extrapolated(L) per quantile.

We compare:
  - 'global'   : single offset per quantile (current method)
  - 'bucket'   : per-length offset, but extrapolate by repeating
                 the largest training length's offset (n=16) for
                 all OOD lengths
  - 'logfit'   : linear extrapolation in log(L)
  - 'linfit'   : linear extrapolation in L

Output: results/calibration/length_conformal_summary.csv
        results/figs/length_cond_conformal.{pdf,png}
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import make_model
from scripts.eval_calibrated import load_npz, predict

DEFAULT_MODELS = {
    'fidelityno_large': 'FidelityNO (5M)',
    'fidelityno':       'FidelityNO (1M)',
    'gnn':              'Generic-GNN',
    'bidir':            'Bidir Trans.',
    'mlp':              'Flat MLP',
}
import os
# Allow CLI override: MODELS="name1@ckpt_dir1,name2@ckpt_dir2"
_models_env = os.environ.get('CONFORMAL_MODELS', '')
_data_env   = os.environ.get('CONFORMAL_DATA',   '')
_out_env    = os.environ.get('CONFORMAL_OUT',    '')

if _models_env:
    MODELS = {}
    MODEL_CKPT_DIR = {}
    for spec in _models_env.split(','):
        spec = spec.strip()
        if '@' in spec:
            n, d = spec.split('@', 1)
        else:
            n, d = spec, 'checkpoints'
        MODELS[n] = n
        MODEL_CKPT_DIR[n] = d
else:
    MODELS = DEFAULT_MODELS
    MODEL_CKPT_DIR = {n: 'checkpoints' for n in MODELS}

SEEDS = [int(s) for s in os.environ.get('CONFORMAL_SEEDS', '0 1 2 3 4').split()]
DATA = Path(_data_env) if _data_env else (ROOT / 'data')
CKPT = ROOT / 'checkpoints'  # legacy fallback
OUT_DIR = Path(_out_env) if _out_env else (ROOT / 'results' / 'calibration')
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / 'results' / 'figs'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def per_length_offsets(q, y, lengths, levels):
    """Returns dict {L: offsets[Q]} on the points in (q, y) per length L."""
    out = {}
    for L in sorted(set(lengths.tolist())):
        m = lengths == L
        if m.sum() < 30:
            continue
        res = y[m, None] - q[m]
        out[int(L)] = np.array([np.quantile(res[:, j], levels[j])
                                 for j in range(len(levels))])
    return out


def extrapolate(offsets_by_L, target_L, mode='logfit'):
    """offsets_by_L: dict {L: offsets[Q]}. Returns offsets[Q] at target_L."""
    Ls = np.array(sorted(offsets_by_L.keys()), dtype=float)
    O  = np.stack([offsets_by_L[int(L)] for L in Ls])  # [n_train_L, Q]
    if mode == 'global':
        return O.mean(0)
    if mode == 'bucket':
        L_max = int(Ls.max())
        return offsets_by_L[L_max]
    if mode == 'logfit':
        x_train = np.log(Ls); x_t = np.log(target_L)
    elif mode == 'linfit':
        x_train = Ls; x_t = float(target_L)
    else:
        raise ValueError(mode)
    out = np.zeros(O.shape[1])
    for j in range(O.shape[1]):
        beta, alpha = np.polyfit(x_train, O[:, j], 1)
        out[j] = alpha + beta * x_t
    return out


def apply(q, off):
    q2 = np.clip(q + off[None, :], 0.0, 1.0)
    return np.maximum.accumulate(q2, axis=1)


def ece_q(q, y, levels):
    cov = (y[:, None] <= q).mean(0)
    return float(np.abs(cov - levels).mean())


def main():
    rows = []
    for name in MODELS:
        for seed in SEEDS:
            ckpt_dir = Path(MODEL_CKPT_DIR.get(name, str(CKPT)))
            if not ckpt_dir.is_absolute():
                ckpt_dir = ROOT / ckpt_dir
            cp = ckpt_dir / f'{name}_seed{seed}.pt'
            if not cp.exists():
                continue
            ck = torch.load(cp, map_location='cpu')
            cfg = OmegaConf.create(ck['cfg'])
            levels = np.asarray(cfg.model.quantiles, dtype=float)
            raw_id, ds_id = load_npz(DATA / 'id_test.npz')
            input_dim = raw_id['x'].shape[-1]; max_len = raw_id['x'].shape[1]
            model = make_model(cfg.model.name, input_dim, max_len, cfg)
            model.load_state_dict(ck['model']); model.eval()

            # split id_test 50/50 into calibration vs eval
            n = len(ds_id); rng = np.random.default_rng(123 + seed)
            perm = rng.permutation(n); n_cal = n // 2
            cal_idx, eval_idx = perm[:n_cal], perm[n_cal:]

            q_cal, y_cal, _ = predict(model, Subset(ds_id, cal_idx))
            L_cal = raw_id['length'][cal_idx]
            offsets_per_L = per_length_offsets(q_cal, y_cal, L_cal, levels)

            # length-OOD predictions
            raw_l, ds_l = load_npz(DATA / 'length_ood.npz')
            q_l, y_l, _ = predict(model, ds_l)
            L_l = raw_l['length']

            for mode in ['global', 'bucket', 'logfit', 'linfit']:
                # apply per-length offset
                q_l_cal = np.copy(q_l)
                for L_target in sorted(set(L_l.tolist())):
                    m = L_l == L_target
                    off = extrapolate(offsets_per_L, int(L_target), mode=mode)
                    q_l_cal[m] = apply(q_l[m], off)
                # report per-length
                for L_target in sorted(set(L_l.tolist())):
                    m = L_l == L_target
                    mae = float(np.abs(q_l_cal[m].mean(1) - y_l[m]).mean())
                    ece = ece_q(q_l_cal[m], y_l[m], levels)
                    rows.append(dict(model=name, seed=seed, mode=mode,
                                     length=int(L_target), mae=mae, ece=ece))
            print(f'done {name} seed={seed}')
    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / 'length_conformal_summary.csv'
    df.to_csv(out_csv, index=False)
    print(f'wrote {out_csv}, rows={len(df)}')

    # aggregate
    agg = (df.groupby(['model', 'mode', 'length'])
             .agg(mae_mean=('mae', 'mean'), mae_std=('mae', 'std'),
                  ece_mean=('ece', 'mean'), ece_std=('ece', 'std'))
             .reset_index())
    agg.to_csv(OUT_DIR / 'length_conformal_aggregate.csv', index=False)
    print(agg.to_string())

    # plot ECE vs length per mode (averaged over models)
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.4))
    mode_colors = {'global': '#7f7f7f', 'bucket': '#ff7f0e',
                   'logfit': '#1f77b4', 'linfit': '#2ca02c'}
    g = (agg.groupby(['mode', 'length'])
            .agg(ece=('ece_mean', 'mean'), mae=('mae_mean', 'mean'))
            .reset_index())
    for m in mode_colors:
        sub = g[g['mode'] == m].sort_values('length')
        ax[0].plot(sub['length'], sub['ece'], '-o', color=mode_colors[m],
                   label=m, lw=1.2, ms=5)
        ax[1].plot(sub['length'], sub['mae'], '-o', color=mode_colors[m],
                   label=m, lw=1.2, ms=5)
    ax[0].set_xlabel('Sequence length $n$'); ax[0].set_ylabel('Quantile ECE')
    ax[0].set_title('Length-OOD calibration vs conformal mode')
    ax[0].grid(True, alpha=0.3, lw=0.5); ax[0].legend(fontsize=8, frameon=False)
    ax[1].set_xlabel('Sequence length $n$'); ax[1].set_ylabel('Mean MAE')
    ax[1].set_title('Length-OOD MAE vs conformal mode')
    ax[1].grid(True, alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'length_cond_conformal.pdf')
    fig.savefig(FIG_DIR / 'length_cond_conformal.png', dpi=160)
    print('wrote length_cond_conformal.pdf')


if __name__ == '__main__':
    main()
