"""Measure affine OOD calibration when calibration labels are finite-shot DFE.

The main calibration experiments use exact labels. This audit replaces those
labels with budget-matched stratified DFE estimates and reports the downstream
test MAE plus the total offline quantum-shot cost.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from physics.baselines.dfe import direct_fidelity_estimate
from physics.channels.base import Channel
from scripts.eval_recalibrated import fit_recalibrate, predict_means


def true_channel(raw: np.lib.npyio.NpzFile, idx: int) -> Channel:
    choi = raw["true_choi_real"][idx].astype(np.float64)
    choi = choi + 1j * raw["true_choi_imag"][idx].astype(np.float64)
    dim = int(round(np.sqrt(choi.shape[0])))
    return Channel("true", dim, choi=choi)


def noisy_labels(
    raw: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    total_shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = np.empty(len(indices), dtype=np.float64)
    for j, idx in enumerate(indices):
        result = direct_fidelity_estimate(
            [true_channel(raw, int(idx))],
            num_paulis=total_shots,
            M_per_pauli=1,
            noise="finite",
            strategy="stratified",
            total_shots=total_shots,
            rng=rng,
        )
        labels[j] = np.clip(result.F_hat, 0.0, 1.0)
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--calib-sizes", default="16,32,64,128")
    ap.add_argument("--shots-per-label", default="16,32,64,128,512")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--split-seed", type=int, default=20260806)
    ap.add_argument("--out", default="results_mlst/noisy_ood_calibration.csv")
    args = ap.parse_args()

    raw = np.load(args.data, allow_pickle=True)
    y = raw["y"].astype(np.float64)
    calib_sizes = [int(v) for v in args.calib_sizes.split(",") if v.strip()]
    shot_budgets = [int(v) for v in args.shots_per_label.split(",") if v.strip()]
    max_cal = max(calib_sizes)
    perm = np.random.default_rng(args.split_seed).permutation(len(y))
    cal_pool, test = perm[:max_cal], perm[max_cal:]

    predictions: dict[str, np.ndarray] = {}
    for ckpt in args.ckpts:
        pred, y_ckpt = predict_means(ckpt, args.data)
        if not np.allclose(y, y_ckpt):
            raise RuntimeError(f"ground-truth mismatch for {ckpt}")
        predictions[Path(ckpt).stem] = pred.astype(np.float64)

    rows: list[dict] = []
    for shots in shot_budgets:
        for repeat in range(args.repeats):
            rng = np.random.default_rng(args.split_seed + 100003 * shots + repeat)
            labels = noisy_labels(raw, cal_pool, shots, rng)
            for n_cal in calib_sizes:
                cal = cal_pool[:n_cal]
                noisy = labels[:n_cal]
                for model, pred in predictions.items():
                    a_noisy, b_noisy = fit_recalibrate(pred[cal], noisy, mode="linear")
                    a_exact, b_exact = fit_recalibrate(pred[cal], y[cal], mode="linear")
                    mae_noisy = np.abs(np.clip(a_noisy * pred[test] + b_noisy, 0, 1) - y[test]).mean()
                    mae_exact = np.abs(np.clip(a_exact * pred[test] + b_exact, 0, 1) - y[test]).mean()
                    rows.append(
                        {
                            "model": model,
                            "n_calib": n_cal,
                            "shots_per_label": shots,
                            "total_calibration_shots": n_cal * shots,
                            "repeat": repeat,
                            "mae_noisy_calibration": float(mae_noisy),
                            "mae_exact_calibration": float(mae_exact),
                            "calibration_noise_penalty": float(mae_noisy - mae_exact),
                        }
                    )

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = (
        df.groupby(["model", "n_calib", "shots_per_label", "total_calibration_shots"])
        .agg(
            mae_noisy_mean=("mae_noisy_calibration", "mean"),
            mae_noisy_std=("mae_noisy_calibration", "std"),
            mae_exact=("mae_exact_calibration", "mean"),
            penalty_mean=("calibration_noise_penalty", "mean"),
        )
        .reset_index()
    )
    summary_path = out.with_name(out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"[saved] {out}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
