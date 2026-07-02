#!/usr/bin/env python
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from omegaconf import OmegaConf
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import make_model


def pinball_np(q, y, levels):
    e = y[:, None] - q
    lev = np.asarray(levels)[None, :]
    return float(np.maximum(lev * e, (lev - 1) * e).mean())


def ece_quantile(q, y, levels):
    cov = (y[:, None] <= q).mean(0)
    return float(np.abs(cov - np.asarray(levels)).mean()), cov


def crps_from_quantiles(q, y, levels):
    return float(2 * pinball_np(q, y, levels))


def load_npz(path):
    raw = np.load(path)
    ds = TensorDataset(
        torch.tensor(raw['x']).float(),
        torch.tensor(raw['mask']).float(),
        torch.tensor(raw['y']).float(),
        torch.tensor(raw['stats']).float(),
    )
    return raw, ds


def predict(model, ds, batch_size=256):
    preds, ys = [], []
    t0 = time.perf_counter(); nseq = 0
    with torch.no_grad():
        for x, m, y, stats in DataLoader(ds, batch_size=batch_size):
            q, _ = model(x, m)
            preds.append(q.numpy()); ys.append(y.numpy()); nseq += len(y)
    elapsed = time.perf_counter() - t0
    return np.concatenate(preds), np.concatenate(ys), 1000 * elapsed / max(nseq, 1)


def conformal_offsets(q_cal, y_cal, levels):
    """Additive per-quantile calibration: q'_a(x)=q_a(x)+Quantile_a(y-q_a(x))."""
    levels = np.asarray(levels)
    residuals = y_cal[:, None] - q_cal
    offsets = np.array([np.quantile(residuals[:, j], levels[j]) for j in range(len(levels))], dtype=float)
    return offsets


def apply_offsets(q, offsets):
    q2 = np.clip(q + offsets[None, :], 0.0, 1.0)
    # Keep quantile monotonicity after independent conformal offsets.
    q2 = np.maximum.accumulate(q2, axis=1)
    return q2


def eval_ckpt_calibrated(ckpt_path, data_dir, out_csv, cal_fraction=0.5, cal_seed=123):
    ck = torch.load(ckpt_path, map_location='cpu')
    cfg = OmegaConf.create(ck['cfg'])
    levels = np.asarray(cfg.model.quantiles, dtype=float)

    data_dir = Path(data_dir)
    raw_id, ds_id = load_npz(data_dir / 'id_test.npz')
    input_dim = raw_id['x'].shape[-1]
    max_len = raw_id['x'].shape[1]
    model = make_model(cfg.model.name, input_dim, max_len, cfg)
    model.load_state_dict(ck['model']); model.eval()

    explicit_calib = data_dir / 'calib.npz'
    if explicit_calib.exists():
        _raw_cal, ds_cal = load_npz(explicit_calib)
        q_cal, y_cal, _ = predict(model, ds_cal)
        id_eval_ds = ds_id
        id_eval_idx = None
        calibration_source = 'calib.npz'
        effective_cal_fraction = float(len(ds_cal)) / float(max(len(ds_id), 1))
    else:
        n = len(ds_id)
        rng = np.random.default_rng(cal_seed + int(cfg.seed))
        perm = rng.permutation(n)
        n_cal = max(1, int(round(cal_fraction * n)))
        cal_idx = perm[:n_cal]
        id_eval_idx = perm[n_cal:]
        if len(id_eval_idx) == 0:
            id_eval_idx = cal_idx
        q_cal, y_cal, _ = predict(model, Subset(ds_id, cal_idx))
        id_eval_ds = Subset(ds_id, id_eval_idx)
        calibration_source = 'id_test_fraction'
        effective_cal_fraction = cal_fraction
    offsets = conformal_offsets(q_cal, y_cal, levels)

    rows = []
    split_defs = {
        'id_test_calibrated': (raw_id, id_eval_ds, id_eval_idx),
        'length_ood_calibrated': (*load_npz(data_dir / 'length_ood.npz'), None),
        'family_ood_calibrated': (*load_npz(data_dir / 'family_ood.npz'), None),
    }
    for split, item in split_defs.items():
        raw, ds, idx = item
        q, y, latency = predict(model, ds)
        q = apply_offsets(q, offsets)
        mean = q.mean(1)
        lengths = raw['length'] if idx is None else raw['length'][idx]
        ece, cov = ece_quantile(q, y, levels)
        for L in sorted(set(lengths.tolist())):
            mask = lengths == L
            rows.append({
                'model': f'{cfg.model.name}_calibrated',
                'base_model': cfg.model.name,
                'seed': int(cfg.seed),
                'split': split,
                'length': int(L),
                'mae': float(np.abs(mean[mask] - y[mask]).mean()),
                'pinball': pinball_np(q[mask], y[mask], levels),
                'crps': crps_from_quantiles(q[mask], y[mask], levels),
                'ece': ece,
                'latency_ms': latency,
                'cal_fraction': effective_cal_fraction,
                'calibration_source': calibration_source,
            })
    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cal-fraction', type=float, default=0.5)
    ap.add_argument('--cal-seed', type=int, default=123)
    args = ap.parse_args()
    eval_ckpt_calibrated(args.ckpt, args.data_dir, args.out, args.cal_fraction, args.cal_seed)
