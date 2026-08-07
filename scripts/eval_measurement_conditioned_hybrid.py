"""Evaluate low-shot measurement-conditioned fidelity estimators.

Every hybrid receives the same OOD calibration indices and the same per-query
Pauli shot budget as stratified DFE.  The estimator is a calibrated convex
fusion of a marginal-only prior and an independent DFE pilot measurement.
Exact-label and finite-label calibration are reported separately; the latter
explicitly accounts for both label-characterisation and fusion-fit shots.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from physics.baselines.hybrid import (
    apply_convex_fusion,
    batch_pauli_expectations_from_choi,
    fit_convex_fusion,
    sample_identity_dfe,
)
from scripts.eval_exact_composition import exact_predictions
from scripts.eval_label_budget_baselines import (
    apply_ridge,
    fit_ridge,
    metrics,
    product_predictions,
    summary_features,
)
from scripts.eval_recalibrated import fit_recalibrate, predict_means


def affine_prior(source: np.ndarray, target: np.ndarray, cal: np.ndarray) -> np.ndarray:
    slope, intercept = fit_recalibrate(source[cal], target, mode="linear")
    return np.clip(slope * source + intercept, 0.0, 1.0)


def metric_row(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return metrics(np.clip(prediction, 0.0, 1.0), target)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--budgets", default="4,8,16,32,64")
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=4096)
    ap.add_argument("--label-shots", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=20260806)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--out", default="results_mlst/measurement_conditioned_hybrid.csv")
    args = ap.parse_args()

    budgets = [int(value) for value in args.budgets.split(",") if value.strip()]
    if min(budgets) < 4:
        raise ValueError("single-qubit stratified DFE requires at least four shots")

    raw = np.load(args.data, allow_pickle=True)
    y = raw["y"].astype(np.float64)
    if args.n_calib < 4 or args.n_calib + args.n_test > len(y):
        raise ValueError("invalid calibration/test sizes")
    permutation = np.random.default_rng(args.split_seed).permutation(len(y))
    cal = permutation[: args.n_calib]
    test = permutation[args.n_calib : args.n_calib + args.n_test]

    choi = raw["true_choi_real"].astype(np.float64)
    choi = choi + 1j * raw["true_choi_imag"].astype(np.float64)
    expectations = batch_pauli_expectations_from_choi(choi)
    fidelity_from_paulis = expectations.mean(axis=1)
    max_fidelity_delta = float(np.max(np.abs(fidelity_from_paulis - y)))
    if max_fidelity_delta > 5e-6:
        raise RuntimeError(f"Pauli/label fidelity mismatch: {max_fidelity_delta}")

    exact = exact_predictions(raw)
    product = product_predictions(raw)
    x_summary = summary_features(raw, product, exact)
    neural = {}
    for ckpt in args.ckpts:
        prediction, returned_y = predict_means(ckpt, args.data)
        if not np.allclose(returned_y, y, atol=1e-6):
            raise RuntimeError(f"ground-truth mismatch for {ckpt}")
        neural[Path(ckpt).stem] = prediction.astype(np.float64)

    rows: list[dict] = []
    for repeat in range(args.repeats):
        label_rng = np.random.default_rng(args.split_seed + 10_000_019 * (repeat + 1))
        finite_labels, _ = sample_identity_dfe(
            expectations[cal], args.label_shots, label_rng
        )

        pilot_cache = {}
        for budget in budgets:
            cal_rng = np.random.default_rng(
                args.split_seed + 1_000_003 * budget + 1009 * repeat + 17
            )
            test_rng = np.random.default_rng(
                args.split_seed + 2_000_003 * budget + 1013 * repeat + 29
            )
            dfe_cal, _ = sample_identity_dfe(expectations[cal], budget, cal_rng)
            dfe_test, dfe_sigma = sample_identity_dfe(expectations[test], budget, test_rng)
            pilot_cache[budget] = (dfe_cal, dfe_test)
            rows.append(
                {
                    "method": "stratified_dfe",
                    "prior": "none",
                    "checkpoint": "none",
                    "calibration": "none",
                    "repeat": repeat,
                    "n_calib": 0,
                    "n_test": len(test),
                    "per_query_shots": budget,
                    "offline_quantum_shots": 0,
                    "fusion_weight": 1.0,
                    "mean_oracle_dfe_sigma": float(dfe_sigma.mean()),
                    **metric_row(dfe_test, y[test]),
                }
            )

        for calibration, target_cal, label_cost in [
            ("exact_labels", y[cal], 0),
            ("finite_64shot_labels", finite_labels, args.n_calib * args.label_shots),
        ]:
            priors: dict[tuple[str, str], np.ndarray] = {}
            priors[("exact_marginal_affine", "none")] = affine_prior(exact, target_cal, cal)
            ridge_fit = fit_ridge(x_summary[cal], target_cal, args.ridge)
            priors[("summary_ridge", "none")] = np.clip(
                apply_ridge(x_summary, ridge_fit), 0.0, 1.0
            )
            for checkpoint, source in neural.items():
                priors[("bidir_affine", checkpoint)] = affine_prior(source, target_cal, cal)

            for (prior_name, checkpoint), prior in priors.items():
                rows.append(
                    {
                        "method": "marginal_only_prior",
                        "prior": prior_name,
                        "checkpoint": checkpoint,
                        "calibration": calibration,
                        "repeat": repeat,
                        "n_calib": args.n_calib,
                        "n_test": len(test),
                        "per_query_shots": 0,
                        "offline_quantum_shots": label_cost,
                        "fusion_weight": 0.0,
                        "mean_oracle_dfe_sigma": np.nan,
                        **metric_row(prior[test], y[test]),
                    }
                )
                for budget in budgets:
                    dfe_cal, dfe_test = pilot_cache[budget]
                    weight = fit_convex_fusion(prior[cal], dfe_cal, target_cal)
                    hybrid = np.clip(
                        apply_convex_fusion(prior[test], dfe_test, weight),
                        0.0,
                        1.0,
                    )
                    rows.append(
                        {
                            "method": "measurement_conditioned_hybrid",
                            "prior": prior_name,
                            "checkpoint": checkpoint,
                            "calibration": calibration,
                            "repeat": repeat,
                            "n_calib": args.n_calib,
                            "n_test": len(test),
                            "per_query_shots": budget,
                            "offline_quantum_shots": label_cost + args.n_calib * budget,
                            "fusion_weight": weight,
                            "mean_oracle_dfe_sigma": np.nan,
                            **metric_row(hybrid, y[test]),
                        }
                    )

    result = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    summary = (
        result.groupby(
            ["method", "prior", "calibration", "per_query_shots"], dropna=False
        )
        .agg(
            mae_mean=("mae_F_e", "mean"),
            mae_std=("mae_F_e", "std"),
            rmse_mean=("rmse_F_e", "mean"),
            weight_mean=("fusion_weight", "mean"),
            weight_std=("fusion_weight", "std"),
            offline_shots=("offline_quantum_shots", "max"),
            runs=("mae_F_e", "size"),
        )
        .reset_index()
    )
    summary_path = out.with_name(out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    view = summary[
        (summary["method"] == "stratified_dfe")
        | (
            (summary["calibration"] == "finite_64shot_labels")
            & summary["prior"].isin(["bidir_affine", "summary_ridge"])
        )
    ]
    print(view.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"[max Pauli-label delta] {max_fidelity_delta:.3e}")
    print(f"[saved] {out}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
