"""Eval one ckpt against the length-extreme splits (id_test_short + length_ood)."""
from __future__ import annotations
import argparse, time, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import make_model, prediction_to_quantiles
from eval import pinball_np, ece_quantile, crps_from_quantiles


def load(path):
    d = np.load(path, allow_pickle=True)
    return d, TensorDataset(
        torch.tensor(d['x']).float(),
        torch.tensor(d['mask']).float(),
        torch.tensor(d['y']).float(),
        torch.tensor(d['stats']).float(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = OmegaConf.create(ck['cfg'])
    levels = cfg.model.quantiles
    splits = {
        'id_test_short': str(Path(args.data_dir) / 'id_test_short.npz'),
        'length_ood':    str(Path(args.data_dir) / 'length_ood.npz'),
    }
    sample = np.load(next(iter(splits.values())), allow_pickle=True)
    input_dim = sample['x'].shape[-1]; max_len = sample['x'].shape[1]
    model = make_model(cfg.model.name, input_dim, max_len, cfg)
    model.load_state_dict(ck['model']); model.eval()

    rows = []
    for name, path in splits.items():
        raw, ds = load(path)
        preds, ys = [], []
        t0 = time.perf_counter(); nseq = 0
        with torch.no_grad():
            for x, m, y, stats in DataLoader(ds, batch_size=256):
                pred, _ = model(x, m)
                q = prediction_to_quantiles(pred, torch.tensor(levels, dtype=torch.float32))
                preds.append(q.numpy()); ys.append(y.numpy()); nseq += len(y)
        elapsed = time.perf_counter() - t0
        q = np.concatenate(preds); y = np.concatenate(ys); mean = q.mean(1)
        ece, _ = ece_quantile(q, y, levels)
        for L in sorted(set(raw['length'].tolist())):
            idx = raw['length'] == L
            rows.append({
                'model': cfg.model.name,
                'seed': cfg.seed,
                'split': name,
                'length': int(L),
                'mae': float(np.abs(mean[idx] - y[idx]).mean()),
                'pinball': pinball_np(q[idx], y[idx], levels),
                'crps': crps_from_quantiles(q[idx], y[idx], levels),
                'ece': ece,
                'latency_ms': 1000 * elapsed / max(nseq, 1),
            })
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df)


if __name__ == '__main__':
    main()
