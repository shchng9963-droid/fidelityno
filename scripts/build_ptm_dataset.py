#!/usr/bin/env python
"""Convert Choi-feature collision dataset to PTM-feature collision dataset (vectorised)."""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physics.channels.ptm import _pauli_basis

D = 2
HALF = D ** 4   # 16
FULL = 2 * HALF # 32


def build_T(d):
    """T[i,j,m,p,n,q] = (Pi)[q,p] * (Pj)[m,n] so that
       PTM[N,i,j] = (1/d) Re einsum(ijmpnq,Nmpnq->Nij)(T, C)"""
    paulis = _pauli_basis(d)
    n = len(paulis)
    T = np.zeros((n, n, d, d, d, d), dtype=complex)
    for i, Pi in enumerate(paulis):
        for j, Pj in enumerate(paulis):
            T[i, j] = np.einsum('qp,mn->mpnq', Pi, Pj)
    return T


T_D = {2: build_T(2), 4: build_T(4)}


def convert_x_vec(x: np.ndarray, mask: np.ndarray, d: int = D) -> np.ndarray:
    """x: (N, T_seq, FULL) float32. Returns same-shape PTM feature array."""
    N, T_seq, F = x.shape
    out = np.zeros_like(x)
    real = x[:, :, :HALF].reshape(N, T_seq, d * d, d * d)
    imag = x[:, :, HALF:].reshape(N, T_seq, d * d, d * d)
    choi = (real + 1j * imag).reshape(N * T_seq, d, d, d, d)
    R = np.einsum('ijmpnq,Nmpnq->Nij', T_D[d], choi).real / d  # (N*T_seq, d^2, d^2)
    R = R.reshape(N, T_seq, HALF)
    # Zero-mask invalid timesteps for cleanliness
    mask_b = mask.astype(bool)[:, :, None]
    out[:, :, :HALF] = (R.astype(np.float32) * mask_b)
    return out


def main():
    src_dir = ROOT / 'data' / 'collision'
    dst_dir = ROOT / 'data' / 'collision_ptm'
    dst_dir.mkdir(exist_ok=True, parents=True)
    for name in ['train', 'calib', 'id_test', 'length_ood', 'family_ood']:
        src = src_dir / f'{name}.npz'
        dst = dst_dir / f'{name}.npz'
        if not src.exists():
            print(f'[skip] {src} (not found)')
            continue
        d = np.load(src, allow_pickle=True)
        t0 = time.perf_counter()
        x_ptm = convert_x_vec(d['x'], d['mask'])
        elapsed = time.perf_counter() - t0
        N = d['x'].shape[0]
        print(f'[done] {name}: N={N} in {elapsed:.2f}s')
        new = {k: d[k] for k in d.files}
        new['x'] = x_ptm
        np.savez(dst, **new)
        print(f'       saved {dst}')


if __name__ == '__main__':
    main()
