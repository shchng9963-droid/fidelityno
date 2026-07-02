"""Evaluate the DFE baseline on a cached eval split.

DFE doesn't have a "trained model"; instead, it has a knob (S, M) that
trades quantum-shot budget for fidelity-estimate variance.  We run the
estimator on every sample at several budgets and report MAE for each.

Inputs
------
A cached split produced by ``scripts/gen_data.py`` or
``scripts/gen_real_hardware_data.py`` containing ``x`` (per-channel
features), ``mask``, ``y`` (true F_e), and ``length``.

Outputs
-------
``--out`` CSV with columns:
    backend, n_seq, S, M, mae_F_e, mae_F_avg, sigma_emp,
    quantum_shots_per_seq, n_seq_eval

This is the "QI-native" baseline of P0.3 in PRXQ_PLAN.md.
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

from physics.baselines.dfe import direct_fidelity_estimate
from physics.channels.base import Channel
from physics.fidelity import ef_to_avg


def features_to_choi(feat: np.ndarray, dim: int) -> np.ndarray:
    """Inverse of physics.channels.base.choi_to_real_features.

    feat is shape (2*N^2,) where N = dim*dim, and the layout is
    [Re(C).flatten(), Im(C).flatten()].
    """
    N = dim * dim
    assert feat.size == 2 * N * N, (feat.size, 2 * N * N)
    real = feat[: N * N].reshape(N, N).astype(np.float64)
    imag = feat[N * N :].reshape(N, N).astype(np.float64)
    return real + 1j * imag


def reconstruct_channel_list(x: np.ndarray, mask: np.ndarray, dim: int) -> list[Channel]:
    """Rebuild Channel objects from (max_len, feat_dim) features.

    Channels with mask==0 are skipped.  We only need .choi for DFE
    (Channel internally fills superop/kraus from Choi).
    """
    out = []
    for i in range(x.shape[0]):
        if mask[i] < 0.5:
            break
        choi = features_to_choi(x[i], dim)
        out.append(Channel(name="rec", dim=dim, choi=choi))
    return out


def true_channel_from_npz(raw, idx: int, dim: int) -> Channel | None:
    """If the dataset stores a 'true Choi' (e.g. non-Markovian collision data),
    return a single-channel list containing that Choi.  Otherwise None.
    """
    if "true_choi_real" not in raw.files or "true_choi_imag" not in raw.files:
        return None
    cr = raw["true_choi_real"][idx].astype(np.float64)
    ci = raw["true_choi_imag"][idx].astype(np.float64)
    return Channel(name="true", dim=dim, choi=cr + 1j * ci)


def run_dfe_split(
    npz_path: str,
    *,
    pauli_budgets: list[int],
    M_per_pauli: int,
    n_eval: int,
    seed: int,
    dim: int,
    backend: str,
) -> pd.DataFrame:
    raw = np.load(npz_path, allow_pickle=True)
    n_total = raw["x"].shape[0]
    n_eval = min(n_eval, n_total)

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=n_eval, replace=False)
    rows = []
    for S in pauli_budgets:
        # one common rng per (S) sweep so estimator-variance comparisons
        # are reproducible.
        rng_s = np.random.default_rng(seed * 1000 + S)
        errs_F_e = []
        errs_F_avg = []
        F_hats = []
        F_truths = []
        t0 = time.perf_counter()
        for i in tqdm(indices, desc=f"DFE {backend} S={S}"):
            true_ch = true_channel_from_npz(raw, i, dim)
            if true_ch is not None:
                channels = [true_ch]
            else:
                channels = reconstruct_channel_list(raw["x"][i], raw["mask"][i], dim)
            res = direct_fidelity_estimate(
                channels,
                target_unitary=None,  # noise channels: V = I
                num_paulis=S,
                M_per_pauli=M_per_pauli,
                noise="finite",
                rng=rng_s,
            )
            F_hat = float(np.clip(res.F_hat, 0.0, 1.0))
            F_true = float(raw["y"][i])
            errs_F_e.append(abs(F_hat - F_true))
            errs_F_avg.append(abs(ef_to_avg(F_hat, dim) - ef_to_avg(F_true, dim)))
            F_hats.append(F_hat); F_truths.append(F_true)
        elapsed = time.perf_counter() - t0
        rows.append({
            "backend": backend,
            "n_seq": n_eval,
            "S": int(S),
            "M": int(M_per_pauli),
            "quantum_shots_per_seq": int(S * M_per_pauli),
            "mae_F_e": float(np.mean(errs_F_e)),
            "std_F_e": float(np.std(errs_F_e)),
            "mae_F_avg": float(np.mean(errs_F_avg)),
            "rmse_F_e": float(np.sqrt(np.mean(np.square(errs_F_e)))),
            "elapsed_s_total": elapsed,
            "elapsed_ms_per_seq": 1000 * elapsed / max(n_eval, 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="data/real_hardware",
                    help="Root containing per-backend dirs with real_hw_test.npz")
    ap.add_argument("--data", default=None,
                    help="Single .npz path (alternative to --data-root). "
                         "Treated as --backend=stem.")
    ap.add_argument("--backends", nargs="*", default=None)
    ap.add_argument("--pauli-budgets", default="10,30,100,300,1000")
    ap.add_argument("--M", type=int, default=200,
                    help="measurement repetitions per Pauli (default 200)")
    ap.add_argument("--n-eval", type=int, default=512,
                    help="number of test sequences to evaluate per split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--out", default="results_prxq/dfe/dfe_summary.csv")
    args = ap.parse_args()

    pauli_budgets = [int(s) for s in args.pauli_budgets.split(",") if s.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    if args.data is not None:
        backend = Path(args.data).stem
        df = run_dfe_split(args.data, pauli_budgets=pauli_budgets, M_per_pauli=args.M,
                           n_eval=args.n_eval, seed=args.seed, dim=args.dim,
                           backend=backend)
        all_dfs.append(df)
    else:
        root = Path(args.data_root)
        if args.backends is None:
            backends = sorted(d.name for d in root.iterdir() if d.is_dir())
        else:
            backends = list(args.backends)
        for be in backends:
            npz = root / be / "real_hw_test.npz"
            if not npz.exists():
                print(f"[skip] {be}: no real_hw_test.npz")
                continue
            df = run_dfe_split(str(npz), pauli_budgets=pauli_budgets, M_per_pauli=args.M,
                               n_eval=args.n_eval, seed=args.seed, dim=args.dim,
                               backend=be)
            all_dfs.append(df)

    if not all_dfs:
        print("[error] nothing to evaluate.")
        return
    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"\n[saved] {out_path}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
