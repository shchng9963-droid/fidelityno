"""Analytic + Monte-Carlo baselines for the real-hardware FidelityNO eval.

Reads each ``data/real_hardware/<backend>/real_hw_test.npz`` and computes:
  - product_bound:        F = ∏ F_i
  - fvg_bound:            F = max(0, 1 - Σ (1 - F_i))   (Fuchs–van de Graaf)
  - diamond_telescope:    F = max(0, 1 - Σ √(1 - F_i^2))
  - analytic_best:        max of the above
  - mc_K                  for K ∈ {10, 100, 1000}: importance-sampled
                          Kraus-path Monte Carlo of the F_e formula
                          (using ``scripts.eval_mc.mc_process_fidelity``).

Outputs a single CSV at results_prxq/real_hardware/analytic_baselines.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from physics.fidelity import ef_to_avg
from scripts.eval_mc import features_to_choi, kraus_from_choi, mc_process_fidelity, infer_dim_from_feature_dim


LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def metrics_point(pred: np.ndarray, y: np.ndarray, length: np.ndarray, model: str, backend: str, latency_ms: float) -> list[dict]:
    rows = []
    q = np.repeat(pred[:, None], len(LEVELS), axis=1)
    cov = (y[:, None] <= q).mean(0)
    ece = float(np.abs(cov - LEVELS).mean())
    e = y[:, None] - q
    pin = float(np.maximum(LEVELS[None, :] * e, (LEVELS[None, :] - 1) * e).mean())
    overall = {
        "backend": backend,
        "model": model,
        "n": int(len(y)),
        "mae_F_e": float(np.abs(pred - y).mean()),
        "mae_F_avg": float(np.abs(np.array([ef_to_avg(p, 2) for p in pred]) - np.array([ef_to_avg(yi, 2) for yi in y])).mean()),
        "max_abs_err_F_e": float(np.abs(pred - y).max()),
        "ece_point": ece,
        "pinball": pin,
        "crps": 2 * pin,
        "latency_ms": latency_ms,
    }
    rows.append({**overall, "length": -1})  # -1 = pooled
    for L in sorted(set(length.tolist())):
        idx = length == L
        if idx.sum() == 0:
            continue
        rows.append({
            **overall,
            "length": int(L),
            "n": int(idx.sum()),
            "mae_F_e": float(np.abs(pred[idx] - y[idx]).mean()),
            "max_abs_err_F_e": float(np.abs(pred[idx] - y[idx]).max()),
        })
    return rows


def analytic_predictions(d: dict) -> dict[str, np.ndarray]:
    pf = d["per_fid"]
    mask = d["mask"]
    prod = np.prod(np.where(mask > 0, pf, 1.0), axis=1)
    fvg = np.clip(1.0 - np.sum(np.where(mask > 0, 1 - pf, 0.0), axis=1), 0, 1)
    sqrt_term = np.sqrt(np.clip(1 - pf ** 2, 0, None))
    diamond_lb = np.clip(1.0 - np.sum(np.where(mask > 0, sqrt_term, 0.0), axis=1), 0, 1)
    return {
        "product_bound": prod,
        "fvg_bound": fvg,
        "diamond_telescope": diamond_lb,
        "analytic_best": np.maximum(diamond_lb, prod),
    }


def mc_predictions_for_split(npz_path: Path, ks: list[int], rng: np.random.Generator) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Return one prediction array per K, plus per-K wall-time."""
    d = np.load(npz_path, allow_pickle=True)
    x = d["x"]; mask = d["mask"]
    n, max_len, feat_dim = x.shape
    dim = infer_dim_from_feature_dim(feat_dim)

    # Convert each feature row into a Kraus list, once.
    kraus_seqs: list[list[list[np.ndarray]]] = []
    for i in range(n):
        L = int(mask[i].sum())
        seq = []
        for j in range(L):
            choi = features_to_choi(x[i, j])
            seq.append(kraus_from_choi(choi, dim))
        kraus_seqs.append(seq)

    preds = {k: np.zeros(n, dtype=np.float64) for k in ks}
    wall = {k: 0.0 for k in ks}
    for k in ks:
        t0 = time.perf_counter()
        for i in range(n):
            preds[k][i] = mc_process_fidelity(kraus_seqs[i], dim, k, rng)
        wall[k] = time.perf_counter() - t0
    return preds, wall


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="data/real_hardware")
    ap.add_argument("--out", default="results_prxq/real_hardware/analytic_baselines.csv")
    ap.add_argument("--mc-budgets", default="10,100,1000")
    ap.add_argument("--mc-subset", type=int, default=0,
                    help="If >0, evaluate MC on only the first N sequences "
                         "per backend (cheaper).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    backends = sorted([d for d in data_root.iterdir() if d.is_dir()])
    for be_dir in backends:
        backend = be_dir.name
        npz_path = be_dir / "real_hw_test.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path, allow_pickle=True)
        y = d["y"]; length = d["length"]
        # analytic baselines
        t0 = time.perf_counter()
        preds_an = analytic_predictions(d)
        elapsed = (time.perf_counter() - t0) / max(len(y), 1) * 1000
        for name, pred in preds_an.items():
            rows.extend(metrics_point(pred, y, length, name, backend, latency_ms=elapsed))

        # Monte-Carlo
        ks = [int(x) for x in args.mc_budgets.split(",")]
        if args.mc_subset > 0:
            n_sub = min(args.mc_subset, len(y))
            # build a temporary npz-like dict on a subset
            d_sub_path = be_dir / f"_mc_subset_{n_sub}.npz"
            np.savez_compressed(d_sub_path,
                                 x=d["x"][:n_sub], mask=d["mask"][:n_sub],
                                 y=d["y"][:n_sub], length=d["length"][:n_sub])
            mc_npz_path = d_sub_path
            y_mc = y[:n_sub]; length_mc = length[:n_sub]
        else:
            mc_npz_path = npz_path
            y_mc = y; length_mc = length

        mc_preds, mc_walls = mc_predictions_for_split(mc_npz_path, ks, rng)
        for k, pred in mc_preds.items():
            rows.extend(metrics_point(pred, y_mc, length_mc, f"mc_K{k}", backend,
                                       latency_ms=1000 * mc_walls[k] / max(len(y_mc), 1)))

        if args.mc_subset > 0:
            d_sub_path.unlink(missing_ok=True)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    df = pd.DataFrame(rows)
    pooled = df[df.length == -1]
    print(pooled.to_string(index=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
