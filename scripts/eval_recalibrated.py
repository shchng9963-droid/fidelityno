"""Post-hoc affine recalibration baseline for FidelityNO predictions.

Given (a) raw quantile predictions and (b) labeled OOD samples (or a
held-out *per-OOD-split calibration* set), fit a 1D linear regression
F_corr = a * F_raw + b on those samples and report the corrected MAE
on the rest of the split.

Why: under distribution shift the model's *ordering* of sequences
typically remains correct (high Pearson correlation) but its absolute
scale drifts toward the training y_mean.  An affine correction with
2 parameters often closes most of the gap to product_bound.

Usage:
  python scripts/eval_recalibrated.py \
      --ckpts checkpoints/collision/fidelityno_seed{0..4}.pt \
      --data-root data/collision \
      --calib-frac 0.1 \
      --out results_prxq/collision/recalibrated.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from train import make_model, prediction_to_quantiles


def predict_quantiles(ckpt_path: str, npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return all predictive quantiles and targets for one checkpoint."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["cfg"])
    levels = list(cfg.model.quantiles)
    raw = np.load(npz_path, allow_pickle=True)
    model = make_model(cfg.model.name, raw["x"].shape[-1], raw["x"].shape[1], cfg)
    model.load_state_dict(ck["model"]); model.eval()

    ds = TensorDataset(
        torch.tensor(raw["x"]).float(),
        torch.tensor(raw["mask"]).float(),
        torch.tensor(raw["y"]).float(),
        torch.tensor(raw["stats"]).float(),
    )
    preds, ys = [], []
    with torch.no_grad():
        for x, m, y, _ in DataLoader(ds, batch_size=256):
            pred, _ = model(x, m)
            q = prediction_to_quantiles(pred, torch.tensor(levels))
            preds.append(q.numpy()); ys.append(y.numpy())
    q = np.concatenate(preds)
    y = np.concatenate(ys)
    return q, y


def predict_means(ckpt_path: str, npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    q, y = predict_quantiles(ckpt_path, npz_path)
    return q.mean(axis=1), y


def fit_recalibrate(pred: np.ndarray, y: np.ndarray, mode: str = "linear") -> tuple[float, float]:
    if mode == "bias":
        return 1.0, float((y - pred).mean())
    A = np.vstack([pred, np.ones_like(pred)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data-root", default="data/collision")
    ap.add_argument("--splits", nargs="+", default=["id_test", "length_ood", "family_ood"])
    ap.add_argument("--calib-frac", type=float, default=0.1,
                    help="fraction of OOD samples used for affine fit")
    ap.add_argument("--mode", default="linear", choices=["linear", "bias"])
    ap.add_argument("--out", default="results_prxq/collision/recalibrated.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for ckpt in args.ckpts:
        for split in args.splits:
            npz = f"{args.data_root}/{split}.npz"
            mean, y = predict_means(ckpt, npz)
            n = len(y)
            n_calib = max(2, int(round(args.calib_frac * n)))
            idx = rng.permutation(n)
            calib_idx = idx[:n_calib]
            test_idx = idx[n_calib:]
            a, b = fit_recalibrate(mean[calib_idx], y[calib_idx], mode=args.mode)
            corrected = a * mean[test_idx] + b
            mae_raw = float(np.abs(mean[test_idx] - y[test_idx]).mean())
            mae_corr = float(np.abs(corrected - y[test_idx]).mean())
            rows.append({
                "ckpt": Path(ckpt).name,
                "split": split,
                "n_calib": int(n_calib),
                "n_test": int(len(test_idx)),
                "mode": args.mode,
                "a": a,
                "b": b,
                "mae_F_e_raw": mae_raw,
                "mae_F_e_corr": mae_corr,
                "improvement": mae_raw - mae_corr,
            })
            print(f"{Path(ckpt).name:30s} {split:14s} mode={args.mode}  raw={mae_raw:.4f}  corr={mae_corr:.4f}  (a={a:+.3f}, b={b:+.3f})")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}")
    summary = df.groupby(["split", "mode"]).agg(
        mae_raw_mean=("mae_F_e_raw", "mean"),
        mae_corr_mean=("mae_F_e_corr", "mean"),
        mae_corr_std=("mae_F_e_corr", "std"),
    ).reset_index()
    print("\n=== pooled (across ckpts) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
