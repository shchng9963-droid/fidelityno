"""Stratified 90/10 split of a generated train.npz into train_split/val_split.

Parameterized version of scripts/split_data.py for the family-OOD sweep:
  python scripts/split_data_holdout.py --indir data/family_ood/holdout_amp_damp
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indir', required=True, help='Directory containing train.npz')
    ap.add_argument('--frac', type=float, default=0.9)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    indir = Path(args.indir)
    src = indir / 'train.npz'
    if not src.exists():
        raise FileNotFoundError(src)

    print(f'Loading {src} ...')
    d = np.load(src, allow_pickle=True)
    keys = list(d.keys())
    N = len(d['y'])
    print(f'  {N} samples, keys={keys}')

    lengths = d['length']
    rng = np.random.RandomState(args.seed)
    train_idx, val_idx = [], []
    for L in np.unique(lengths):
        mask = np.where(lengths == L)[0]
        rng.shuffle(mask)
        split = int(args.frac * len(mask))
        train_idx.extend(mask[:split].tolist())
        val_idx.extend(mask[split:].tolist())
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    print(f'  Train: {len(train_idx)}, Val: {len(val_idx)}')

    sample_keys = [k for k in keys if d[k].shape[0] == N]
    meta_keys = [k for k in keys if d[k].shape[0] != N]
    print(f'  sample_keys={sample_keys}  meta_keys={meta_keys}')

    train_d = {k: d[k][train_idx] for k in sample_keys}
    train_d.update({k: d[k] for k in meta_keys})
    val_d = {k: d[k][val_idx] for k in sample_keys}
    val_d.update({k: d[k] for k in meta_keys})

    np.savez_compressed(indir / 'train_split.npz', **train_d)
    np.savez_compressed(indir / 'val_split.npz', **val_d)
    print(f'Saved {indir}/train_split.npz, {indir}/val_split.npz')


if __name__ == '__main__':
    main()
