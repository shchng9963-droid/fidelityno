"""C2: Build the extreme length-OOD probe dataset.

Filter the main train/val splits down to sequences of length <= 4.
Test split = the existing length_ood.npz (L in {24, 32, 48}).

This stresses every architecture's ability to extrapolate the COMPOSITION
operator far beyond seen sequence lengths (12x train length).
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

OUT = Path('data/length_extreme')
OUT.mkdir(parents=True, exist_ok=True)

def filter_le(src: str, dst: str, max_len: int):
    d = np.load(src, allow_pickle=True)
    keep = d['length'] <= max_len
    out = {k: (d[k][keep] if d[k].shape and d[k].shape[0] == len(keep) else d[k]) for k in d.files}
    np.savez_compressed(dst, **out)
    lengths = sorted(set(out['length'].tolist()))
    counts = {int(L): int((out['length'] == L).sum()) for L in lengths}
    print(f'{dst}: n={int(keep.sum())}  lengths={counts}')

filter_le('data/train_split.npz', OUT / 'train.npz', 4)
filter_le('data/val_split.npz',   OUT / 'val.npz',   4)
# Reuse length_ood for the test split (no filter)
import shutil
shutil.copy('data/length_ood.npz', OUT / 'length_ood.npz')
shutil.copy('data/id_test.npz',   OUT / 'id_test_full.npz')
# Also build an "id_test_short" so we can sanity-check the train regime
d = np.load('data/id_test.npz', allow_pickle=True)
keep = d['length'] <= 4
out = {k: (d[k][keep] if d[k].shape and d[k].shape[0] == len(keep) else d[k]) for k in d.files}
np.savez_compressed(OUT / 'id_test_short.npz', **out)
print(f'{OUT}/id_test_short.npz: n={int(keep.sum())}')
print(f'\nC2 dataset ready under {OUT}/')
