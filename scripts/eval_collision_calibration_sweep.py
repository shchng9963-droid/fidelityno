from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from scripts.eval_recalibrated import fit_recalibrate, predict_means


def load_ckpt_seed(ckpt_path: str) -> int:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["cfg"])
    return int(cfg.seed)


def product_bound(path: str) -> tuple[np.ndarray, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    y = raw["y"].astype(np.float64)
    pf = raw["per_fid"].astype(np.float64)
    mask = raw["mask"] > 0
    pred = np.prod(np.where(mask, pf, 1.0), axis=1)
    return pred, y


def evaluate_one(
    pred_mean: np.ndarray,
    y: np.ndarray,
    product_pred: np.ndarray,
    *,
    n_calib: int,
    split_seed: int,
    mode: str,
) -> dict:
    n = len(y)
    if n_calib >= n:
        raise ValueError(f"n_calib={n_calib} must be < n={n}")
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    calib_idx = perm[:n_calib]
    test_idx = perm[n_calib:]

    a, b = fit_recalibrate(pred_mean[calib_idx], y[calib_idx], mode=mode)
    pred_corr = a * pred_mean[test_idx] + b
    return {
        "n_calib": int(n_calib),
        "n_test": int(len(test_idx)),
        "a": float(a),
        "b": float(b),
        "mae_raw": float(np.abs(pred_mean[test_idx] - y[test_idx]).mean()),
        "mae_corr": float(np.abs(pred_corr - y[test_idx]).mean()),
        "mae_product": float(np.abs(product_pred[test_idx] - y[test_idx]).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpts",
        nargs="+",
        default=[str(p) for p in sorted(Path("checkpoints/collision").glob("fidelityno_seed*.pt"))],
    )
    ap.add_argument("--data", default="data/collision/family_ood.npz")
    ap.add_argument("--calib-sizes", default="8,16,32,64,128,256,512,800")
    ap.add_argument("--mode", default="linear", choices=["linear", "bias"])
    ap.add_argument("--split-seed-base", type=int, default=20260614)
    ap.add_argument("--out-csv", default="results_prxq/collision/calibration_sweep.csv")
    ap.add_argument("--out-fig", default="results_prxq/collision/calibration_sweep.pdf")
    args = ap.parse_args()

    calib_sizes = [int(x) for x in args.calib_sizes.split(",") if x.strip()]
    product_pred, y_ref = product_bound(args.data)

    rows: list[dict] = []
    cached_preds: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for ckpt in args.ckpts:
        pred_mean, y = predict_means(ckpt, args.data)
        model_seed = load_ckpt_seed(ckpt)
        if not np.allclose(y, y_ref):
            raise RuntimeError(f"ground truth mismatch for {ckpt}")
        cached_preds[ckpt] = (pred_mean.astype(np.float64), y.astype(np.float64), model_seed)

    for ckpt, (pred_mean, y, model_seed) in cached_preds.items():
        for n_calib in calib_sizes:
            stats = evaluate_one(
                pred_mean,
                y,
                product_pred,
                n_calib=n_calib,
                split_seed=args.split_seed_base + model_seed,
                mode=args.mode,
            )
            rows.append(
                {
                    "ckpt": Path(ckpt).name,
                    "model_seed": int(model_seed),
                    "split_seed": int(args.split_seed_base + model_seed),
                    **stats,
                }
            )
            print(
                f"{Path(ckpt).name:24s} n_cal={n_calib:4d} "
                f"raw={stats['mae_raw']:.4f} corr={stats['mae_corr']:.4f} product={stats['mae_product']:.4f}"
            )

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    agg = (
        df.groupby("n_calib")
        .agg(
            mae_corr_mean=("mae_corr", "mean"),
            mae_corr_std=("mae_corr", "std"),
            mae_raw_mean=("mae_raw", "mean"),
            mae_raw_std=("mae_raw", "std"),
            mae_product_mean=("mae_product", "mean"),
            mae_product_std=("mae_product", "std"),
        )
        .reset_index()
        .sort_values("n_calib")
    )
    agg_path = out_csv.with_name(out_csv.stem + "_aggregate.csv")
    agg.to_csv(agg_path, index=False)

    x = agg["n_calib"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(x, agg["mae_corr_mean"], "-o", lw=2.0, ms=5, color="#1f77b4", label="FidelityNO + affine cal")
    ax.fill_between(
        x,
        agg["mae_corr_mean"] - agg["mae_corr_std"].fillna(0.0),
        agg["mae_corr_mean"] + agg["mae_corr_std"].fillna(0.0),
        color="#1f77b4",
        alpha=0.18,
        linewidth=0,
    )
    ax.axhline(float(agg["mae_raw_mean"].mean()), color="#d62728", ls="--", lw=1.7, label="FidelityNO raw")
    ax.axhline(float(agg["mae_product_mean"].mean()), color="#2ca02c", ls=":", lw=2.0, label="Product bound")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Labelled OOD calibration set size $N_{cal}$")
    ax.set_ylabel("MAE on family-OOD remainder")
    ax.set_title("Collision regime calibration-size sweep")
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig)
    png_path = out_fig.with_suffix(".png")
    fig.savefig(png_path, dpi=170)
    print(f"[saved] {out_csv}")
    print(f"[saved] {agg_path}")
    print(f"[saved] {out_fig}")
    print(f"[saved] {png_path}")
    print("\n=== aggregate ===")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
