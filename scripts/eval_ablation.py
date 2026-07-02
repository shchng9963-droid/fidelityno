#!/usr/bin/env python
"""Eval ablation checkpoints on collision data and produce per-variant pooled summary.

Outputs:
  results_prxq/ablation/per_run.csv     # one row per (variant, seed, split)
  results_prxq/ablation/pooled.csv      # one row per (variant, split), mean over seeds
"""
from __future__ import annotations
import sys, glob, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path('/home/wangshuchang/fidelityno_prxq')
sys.path.insert(0, str(ROOT))
from train import make_model, prediction_to_quantiles, mean_from_prediction


def load_npz(path):
    raw = np.load(path)
    ds = TensorDataset(
        torch.tensor(raw['x']).float(),
        torch.tensor(raw['mask']).float(),
        torch.tensor(raw['y']).float(),
        torch.tensor(raw['stats']).float(),
    )
    return raw, ds


def predict(model, ds, levels):
    levels_t = torch.tensor(levels, dtype=torch.float32)
    preds_q, preds_mean, ys = [], [], []
    with torch.no_grad():
        for x, m, y, _ in DataLoader(ds, batch_size=256):
            out, _ = model(x, m)
            q = prediction_to_quantiles(out, levels_t)
            mu = mean_from_prediction(out)
            preds_q.append(q.numpy()); preds_mean.append(mu.numpy()); ys.append(y.numpy())
    return np.concatenate(preds_q), np.concatenate(preds_mean), np.concatenate(ys)


def ece_quantile(q, y, levels):
    cov = (y[:, None] <= q).mean(0)
    return float(np.abs(cov - np.asarray(levels)).mean())


def pinball_np(q, y, levels):
    e = y[:, None] - q
    lev = np.asarray(levels)[None, :]
    return float(np.maximum(lev * e, (lev - 1) * e).mean())


def eval_one(ckpt_path, data_dir):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = OmegaConf.create(ck['cfg'])
    levels = list(cfg.model.quantiles)
    raw, ds_id = load_npz(data_dir / 'id_test.npz')
    input_dim = raw['x'].shape[-1]; max_len = raw['x'].shape[1]
    model = make_model(cfg.model.name, input_dim, max_len, cfg)
    model.load_state_dict(ck['model']); model.eval()

    rows = []
    for split_name in ['id_test', 'length_ood', 'family_ood']:
        _, ds = load_npz(data_dir / f'{split_name}.npz')
        q, mu, y = predict(model, ds, levels)
        rows.append({
            'split': split_name,
            'mae_mean': float(np.abs(mu - y).mean()),
            'mae_q05': float(np.abs(q[:, len(levels) // 2] - y).mean()),
            'pinball': pinball_np(q, y, levels),
            'crps': 2 * pinball_np(q, y, levels),
            'ece': ece_quantile(q, y, levels),
        })
    return rows


def main():
    data_dir_choi = ROOT / 'data' / 'collision'
    data_dir_ptm = ROOT / 'data' / 'collision_ptm'
    out_dir = ROOT / 'results_prxq' / 'ablation'
    out_dir.mkdir(exist_ok=True, parents=True)

    rows = []
    # default (existing 5-seed ckpts in checkpoints/collision/fidelityno_seed*.pt) -- Choi data
    for ckpt in sorted(glob.glob(str(ROOT / 'checkpoints' / 'collision' / 'fidelityno_seed*.pt'))):
        seed = int(Path(ckpt).stem.split('seed')[-1])
        for r in eval_one(Path(ckpt), data_dir_choi):
            rows.append({'variant': 'default', 'seed': seed, **r})

    # ablation variants -- pick the right data dir per variant
    for ckpt in sorted(glob.glob(str(ROOT / 'checkpoints' / 'collision_ablation' / '*.pt'))):
        stem = Path(ckpt).stem  # e.g. noaux_seed0 or ptm_seed0
        variant, seed_str = stem.rsplit('_seed', 1)
        data_dir = data_dir_ptm if variant == 'ptm' else data_dir_choi
        for r in eval_one(Path(ckpt), data_dir):
            rows.append({'variant': variant, 'seed': int(seed_str), **r})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'per_run.csv', index=False)
    print('wrote', out_dir / 'per_run.csv', f'({len(df)} rows)')

    pooled = df.groupby(['variant', 'split']).agg(
        mae_mean=('mae_mean', 'mean'),
        mae_std=('mae_mean', 'std'),
        ece=('ece', 'mean'),
        n=('seed', 'count'),
    ).round(4).reset_index()
    pooled.to_csv(out_dir / 'pooled.csv', index=False)
    print(pooled.to_string(index=False))


if __name__ == '__main__':
    main()
