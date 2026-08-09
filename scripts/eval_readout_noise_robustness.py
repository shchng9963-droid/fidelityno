"""Evaluate DFE and hybrid fidelity estimates under symmetric readout noise."""
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
    sample_identity_dfe_readout,
)
from scripts.eval_label_budget_baselines import metrics
from scripts.eval_recalibrated import fit_recalibrate, predict_means


def metric_row(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return metrics(np.clip(prediction, 0.0, 1.0), target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--budgets", default="32,64")
    parser.add_argument("--readout-errors", default="0,0.01,0.03,0.05")
    parser.add_argument("--n-calib", type=int, default=64)
    parser.add_argument("--n-test", type=int, default=4032)
    parser.add_argument("--label-shots", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=20260806)
    parser.add_argument("--out", default="results_mlst/readout_noise_robustness.csv")
    args = parser.parse_args()

    budgets = [int(value) for value in args.budgets.split(",") if value.strip()]
    errors = [
        float(value) for value in args.readout_errors.split(",") if value.strip()
    ]
    if any(value < 4 for value in budgets):
        raise ValueError("DFE budgets must be at least four shots")
    if any(not 0.0 <= value < 0.5 for value in errors):
        raise ValueError("readout errors must lie in [0, 0.5)")

    raw = np.load(args.data, allow_pickle=True)
    y = raw["y"].astype(np.float64)
    permutation = np.random.default_rng(args.split_seed).permutation(len(y))
    if args.n_calib < 4 or args.n_calib + args.n_test > len(y):
        raise ValueError("invalid calibration/test sizes")
    cal = permutation[: args.n_calib]
    test = permutation[args.n_calib : args.n_calib + args.n_test]
    choi = raw["true_choi_real"].astype(np.float64)
    choi = choi + 1j * raw["true_choi_imag"].astype(np.float64)
    expectations = batch_pauli_expectations_from_choi(choi)
    max_label_delta = float(np.max(np.abs(expectations.mean(axis=1) - y)))
    if max_label_delta > 5e-6:
        raise RuntimeError(f"Pauli/label fidelity mismatch: {max_label_delta}")

    neural = {}
    for checkpoint in args.ckpts:
        prediction, returned_y = predict_means(checkpoint, args.data)
        if not np.allclose(returned_y, y, atol=1e-6):
            raise RuntimeError(f"ground-truth mismatch for {checkpoint}")
        neural[Path(checkpoint).stem] = prediction.astype(np.float64)

    rows = []
    label_cost = args.n_calib * args.label_shots
    for repeat in range(args.repeats):
        label_rng = np.random.default_rng(args.split_seed + 10_000_019 * (repeat + 1))
        finite_labels, _ = sample_identity_dfe(
            expectations[cal], args.label_shots, label_rng
        )
        priors = {}
        for checkpoint, source in neural.items():
            slope, intercept = fit_recalibrate(source[cal], finite_labels, mode="linear")
            priors[checkpoint] = np.clip(slope * source + intercept, 0.0, 1.0)

        for budget in budgets:
            for readout_error in errors:
                cal_rng = np.random.default_rng(
                    args.split_seed
                    + 20_000_033 * (repeat + 1)
                    + 1009 * budget
                    + int(round(readout_error * 1_000_000))
                )
                test_rng = np.random.default_rng(
                    args.split_seed
                    + 30_000_077 * (repeat + 1)
                    + 1013 * budget
                    + int(round(readout_error * 1_000_000))
                )
                cal_raw, cal_mitigated, _, _ = sample_identity_dfe_readout(
                    expectations[cal], budget, readout_error, cal_rng
                )
                test_raw, test_mitigated, raw_sigma, mitigated_sigma = (
                    sample_identity_dfe_readout(
                        expectations[test], budget, readout_error, test_rng
                    )
                )

                for mitigation, prediction, sigma in [
                    ("none", test_raw, raw_sigma),
                    ("calibrated_symmetric", test_mitigated, mitigated_sigma),
                ]:
                    rows.append(
                        {
                            "method": "stratified_dfe",
                            "checkpoint": "none",
                            "repeat": repeat,
                            "per_query_shots": budget,
                            "readout_error": readout_error,
                            "mitigation": mitigation,
                            "offline_quantum_shots": 0,
                            "fusion_weight": 1.0,
                            "mean_oracle_sigma": float(sigma.mean()),
                            **metric_row(prediction, y[test]),
                        }
                    )

                for checkpoint, prior in priors.items():
                    for mitigation, cal_measurement, test_measurement in [
                        ("none", cal_raw, test_raw),
                        ("calibrated_symmetric", cal_mitigated, test_mitigated),
                    ]:
                        weight = fit_convex_fusion(
                            prior[cal], cal_measurement, finite_labels
                        )
                        prediction = apply_convex_fusion(
                            prior[test], test_measurement, weight
                        )
                        rows.append(
                            {
                                "method": "measurement_conditioned_hybrid",
                                "checkpoint": checkpoint,
                                "repeat": repeat,
                                "per_query_shots": budget,
                                "readout_error": readout_error,
                                "mitigation": mitigation,
                                "offline_quantum_shots": label_cost
                                + args.n_calib * budget,
                                "fusion_weight": weight,
                                "mean_oracle_sigma": np.nan,
                                **metric_row(prediction, y[test]),
                            }
                        )

    result = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    summary = (
        result.groupby(
            ["method", "mitigation", "per_query_shots", "readout_error"],
            dropna=False,
        )
        .agg(
            mae_mean=("mae_F_e", "mean"),
            mae_std=("mae_F_e", "std"),
            rmse_mean=("rmse_F_e", "mean"),
            weight_mean=("fusion_weight", "mean"),
            offline_shots=("offline_quantum_shots", "max"),
            runs=("mae_F_e", "size"),
        )
        .reset_index()
    )
    clean = summary[summary["readout_error"] == 0][
        ["method", "mitigation", "per_query_shots", "mae_mean"]
    ].rename(columns={"mae_mean": "clean_mae"})
    summary = summary.merge(
        clean,
        on=["method", "mitigation", "per_query_shots"],
        how="left",
    )
    summary["mae_penalty_vs_clean"] = summary["mae_mean"] - summary["clean_mae"]
    summary_path = out.with_name(out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"[max Pauli-label delta] {max_label_delta:.3e}")
    print(f"[saved] {out}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
