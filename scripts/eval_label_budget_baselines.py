"""Compare OOD recalibration methods under the same labelled-data budget.

The script deliberately gives every method the identical calibration indices.
It includes label-only, analytic, low-capacity linear, neural, and oracle-memory
baselines so that improvements cannot be attributed solely to access to OOD
labels or to the latent collision parameter.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scripts.eval_exact_composition import exact_predictions
from scripts.eval_recalibrated import fit_recalibrate, predict_means


def product_predictions(raw: np.lib.npyio.NpzFile) -> np.ndarray:
    return np.prod(np.where(raw["mask"] > 0, raw["per_fid"], 1.0), axis=1).astype(np.float64)


def summary_features(raw: np.lib.npyio.NpzFile, product: np.ndarray, exact: np.ndarray) -> np.ndarray:
    mask = raw["mask"] > 0
    pf = raw["per_fid"].astype(np.float64)
    length = mask.sum(axis=1).astype(np.float64)
    infid = np.where(mask, 1.0 - pf, 0.0)
    denom = np.maximum(length, 1.0)
    mean = infid.sum(axis=1) / denom
    centered = np.where(mask, infid - mean[:, None], 0.0)
    std = np.sqrt(np.square(centered).sum(axis=1) / denom)
    maximum = infid.max(axis=1)
    minimum = np.where(mask, infid, np.inf).min(axis=1)
    return np.column_stack(
        [length, product, exact, infid.sum(axis=1), mean, std, minimum, maximum]
    )


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([z, np.ones(len(z))])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return coef, mean, scale


def apply_ridge(x: np.ndarray, fit: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    coef, mean, scale = fit
    return np.column_stack([(x - mean) / scale, np.ones(len(x))]) @ coef


def metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0)
    err = pred - y
    return {
        "mae_F_e": float(np.abs(err).mean()),
        "rmse_F_e": float(np.sqrt(np.square(err).mean())),
        "bias_F_e": float(err.mean()),
        "pearson": float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 0 else float("nan"),
    }


def affine_predict(source: np.ndarray, y: np.ndarray, cal: np.ndarray, test: np.ndarray) -> np.ndarray:
    a, b = fit_recalibrate(source[cal], y[cal], mode="linear")
    return a * source[test] + b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--calib-sizes", default="8,16,32,64,128,256")
    ap.add_argument("--split-seed", type=int, default=20260806)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--out", default="results_mlst/label_budget_baselines.csv")
    args = ap.parse_args()

    raw = np.load(args.data, allow_pickle=True)
    y = raw["y"].astype(np.float64)
    product = product_predictions(raw)
    exact = exact_predictions(raw)
    x_summary = summary_features(raw, product, exact)
    x_oracle = np.column_stack([x_summary, raw["eta"].astype(np.float64)]) if "eta" in raw.files else None

    neural: dict[str, np.ndarray] = {}
    for ckpt in args.ckpts:
        pred, y_ckpt = predict_means(ckpt, args.data)
        if not np.allclose(y, y_ckpt):
            raise RuntimeError(f"ground-truth mismatch for {ckpt}")
        neural[Path(ckpt).stem] = pred.astype(np.float64)

    perm = np.random.default_rng(args.split_seed).permutation(len(y))
    rows: list[dict] = []
    for n_cal in [int(v) for v in args.calib_sizes.split(",") if v.strip()]:
        if n_cal < 2 or n_cal >= len(y):
            raise ValueError(f"invalid calibration size {n_cal} for n={len(y)}")
        cal, test = perm[:n_cal], perm[n_cal:]
        candidates: dict[str, np.ndarray] = {
            "product_raw": product[test],
            "exact_observed_marginals": exact[test],
            "constant_cal_median": np.full(len(test), np.median(y[cal])),
            "product_affine": affine_predict(product, y, cal, test),
            "exact_marginals_affine": affine_predict(exact, y, cal, test),
            "summary_ridge": apply_ridge(x_summary[test], fit_ridge(x_summary[cal], y[cal], args.ridge)),
        }
        if x_oracle is not None:
            candidates["oracle_eta_ridge"] = apply_ridge(
                x_oracle[test], fit_ridge(x_oracle[cal], y[cal], args.ridge)
            )
        for name, pred in neural.items():
            candidates[f"{name}_raw"] = pred[test]
            candidates[f"{name}_affine"] = affine_predict(pred, y, cal, test)

        for model, pred in candidates.items():
            rows.append(
                {
                    "model": model,
                    "n_calib": n_cal,
                    "n_test": len(test),
                    "split_seed": args.split_seed,
                    "ridge": args.ridge if "ridge" in model else np.nan,
                    **metrics(pred, y[test]),
                }
            )

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.pivot(index="model", columns="n_calib", values="mae_F_e").to_string(float_format=lambda x: f"{x:.5f}"))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
