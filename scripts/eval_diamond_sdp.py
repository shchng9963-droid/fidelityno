"""Evaluate the SDP-based diamond-norm fidelity bound on a dataset.

For each test sample we
  1. Reconstruct the *true* composed channel as a Choi matrix
     (uses true_choi if present, else the marginal-Choi composition).
  2. Solve the diamond norm SDP via qutip.dnorm.
  3. Convert d_norm -> F_LB = 1 - d_norm/2  (FvG-style bound).
  4. Report MAE(F_LB, F_e_true).

This is the "true diamond SDP" baseline (PRXQ P1.6), tightening the
analytic-only telescope bound that is in eval_analytic.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from physics.baselines.diamond_norm import (
    diamond_norm_of_difference,
    fidelity_lower_bound_from_diamond,
)
from physics.channels.base import Channel
from physics.composition import compose_channels


def features_to_choi(feat: np.ndarray, dim: int) -> np.ndarray:
    """Inverse of physics.channels.base.choi_to_real_features.

    feat is shape (2*N^2,) where N = dim*dim, and the layout is
    [Re(C).flatten(), Im(C).flatten()].
    """
    N = dim * dim
    real = feat[: N * N].reshape(N, N).astype(np.float64)
    imag = feat[N * N :].reshape(N, N).astype(np.float64)
    return real + 1j * imag


def channel_choi_for_sample(raw, i: int, dim: int, use_true: bool = False) -> np.ndarray:
    """Return the Choi matrix of the COMPOSED channel for sample i.

    use_true=True : returns the *true* non-Markovian Choi if available
                    (information NN does NOT have; "oracle" baseline).
    use_true=False: returns the marginal-composed Choi (the analytic
                    baseline: same information the NN sees).
    """
    if use_true and "true_choi_real" in raw.files:
        return raw["true_choi_real"][i] + 1j * raw["true_choi_imag"][i]
    # Marginal compose via superop product
    from physics.channels.base import choi_to_superop, superop_to_choi
    x = raw["x"][i]
    mask = raw["mask"][i]
    L = int(mask.sum())
    if L == 0:
        d = dim * dim
        return np.eye(d, dtype=complex) * (1.0 / dim)  # identity Choi
    S = np.eye(dim * dim, dtype=complex)
    for t in range(L):
        choi = features_to_choi(x[t], dim)
        S_t = choi_to_superop(choi, dim)
        S = S_t @ S
    return superop_to_choi(S, dim)


def run_diamond_split(
    npz_path: str,
    n_eval: int,
    seed: int = 0,
    dim: int = 2,
    split: str = "id_test",
    use_true: bool = False,
) -> pd.DataFrame:
    raw = np.load(npz_path, allow_pickle=True)
    n_total = raw["y"].shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=min(n_eval, n_total), replace=False)
    rows = []
    t0 = time.perf_counter()
    for i in tqdm(indices, desc=f"diamond {split}"):
        J = channel_choi_for_sample(raw, int(i), dim, use_true=use_true)
        try:
            d = diamond_norm_of_difference(J, dim)
        except Exception as e:
            d = float("nan")
        F_LB = fidelity_lower_bound_from_diamond(d) if np.isfinite(d) else float("nan")
        rows.append({
            "split": split,
            "idx": int(i),
            "F_e_true": float(raw["y"][int(i)]),
            "F_e_diamond_LB": F_LB,
            "diamond_norm": float(d),
            "use_true_choi": int(use_true),
        })
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    df["abs_err"] = (df.F_e_true - df.F_e_diamond_LB).abs()
    print(f"[{split}] n={len(df)} use_true={use_true} MAE={df.abs_err.mean():.4f} elapsed={elapsed:.1f}s")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--splits", default="id_test,length_ood,family_ood")
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--use-true-choi", action="store_true",
                    help="If set, use the stored true Choi (oracle baseline). "
                         "Default: compose marginals (fair analytic baseline).")
    args = ap.parse_args()

    dfs = []
    for split in args.splits.split(","):
        npz = Path(args.data_dir) / f"{split.strip()}.npz"
        if not npz.exists():
            print(f"[skip] {npz} (missing)")
            continue
        df = run_diamond_split(str(npz), args.n_eval, seed=args.seed, dim=args.dim,
                               split=split.strip(), use_true=args.use_true_choi)
        df.to_csv(Path(args.out).parent / f"diamond_per_sample_{split.strip()}_use_true{int(args.use_true_choi)}.csv", index=False)
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[saved] {args.out}")
    print(out.groupby("split").abs_err.agg(["mean", "std", "count"]).to_string())


if __name__ == "__main__":
    main()
