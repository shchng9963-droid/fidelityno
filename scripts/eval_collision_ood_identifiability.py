"""High-eta identifiability audit with an independent latent eta grid."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from physics.baselines.hybrid import ambiguity_statistics
from physics.channels.collision_counterfactual import (
    collision_fidelity_grid_from_params,
    collision_sequence_from_params,
)
from physics.channels.collision_nonmarkov import collision_sequence
from physics.composition import composed_stats, sequence_features
from scripts.eval_recalibrated import fit_recalibrate, predict_means


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-base", type=int, default=2048)
    ap.add_argument("--n-eta", type=int, default=15)
    ap.add_argument("--eta-min", type=float, default=0.85)
    ap.add_argument("--eta-max", type=float, default=0.99)
    ap.add_argument("--lengths", default="8,16,24")
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--calib-repeats", type=int, default=20)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out-prefix", default="results_mlst/collision_ood_identifiability")
    args = ap.parse_args()

    lengths_supported = np.array(sorted({int(x) for x in args.lengths.split(",") if x.strip()}))
    eta_grid = np.linspace(args.eta_min, args.eta_max, args.n_eta, dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    x = np.zeros((args.n_base, 48, 32), dtype=np.float32)
    mask = np.zeros((args.n_base, 48), dtype=np.float32)
    stats = np.zeros((args.n_base, 2), dtype=np.float32)
    sequence_lengths = np.zeros(args.n_base, dtype=np.int32)
    params = np.full((args.n_base, 48, 3), np.nan, dtype=np.float64)
    fidelity_grid = np.zeros((args.n_base, args.n_eta), dtype=np.float64)
    max_marginal_delta = 0.0

    for index in range(args.n_base):
        length = int(rng.choice(lengths_supported))
        base = collision_sequence(length, eta=0.0, rng=rng)
        features, valid = sequence_features(base.marginals, 48, 2, "choi_hermitian")
        x[index], mask[index] = features, valid
        summary = composed_stats(base.marginals)
        stats[index] = (summary["trace"], summary["purity"])
        sequence_lengths[index] = length
        params[index, :length] = base.params
        fidelity_grid[index] = collision_fidelity_grid_from_params(base.params, eta_grid)
        if index < 64:
            low = collision_sequence_from_params(base.params, eta=float(eta_grid[0]))
            high = collision_sequence_from_params(base.params, eta=float(eta_grid[-1]))
            for left, right in zip(low.marginals, high.marginals):
                max_marginal_delta = max(
                    max_marginal_delta, float(np.max(np.abs(left.choi - right.choi)))
                )
    if max_marginal_delta > 1e-12:
        raise RuntimeError(f"marginals changed across eta: {max_marginal_delta}")

    ambiguity = ambiguity_statistics(fidelity_grid)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    input_path = prefix.with_name(prefix.name + "_inputs.npz")
    np.savez_compressed(
        input_path,
        x=x,
        mask=mask,
        y=ambiguity["conditional_median"].astype(np.float32),
        stats=stats,
        length=sequence_lengths,
        params=params,
        eta_grid=eta_grid,
        fidelity_grid=fidelity_grid,
        parameter_seed=np.array(args.seed),
        eta_independent_of_input=np.array(True),
    )

    per_base = pd.DataFrame(
        {
            "base_index": np.arange(args.n_base),
            "length": sequence_lengths,
            "fidelity_min": ambiguity["lower"],
            "fidelity_max": ambiguity["upper"],
            "ambiguity_diameter": ambiguity["diameter"],
            "minimax_abs_lower_bound": ambiguity["minimax_abs_lower_bound"],
            "bayes_mae_lower_bound": ambiguity["uniform_grid_bayes_mae"],
        }
    )
    per_base_path = prefix.with_name(prefix.name + "_per_base.csv")
    per_base.to_csv(per_base_path, index=False)

    permutation = np.random.default_rng(args.seed + 17).permutation(args.n_base)
    cal = permutation[: args.n_calib]
    test = permutation[args.n_calib :]
    test_ambiguity = ambiguity_statistics(fidelity_grid[test])
    bayes_test = float(test_ambiguity["uniform_grid_bayes_mae"].mean())
    rows = []
    for ckpt in args.ckpts:
        prediction, returned_y = predict_means(ckpt, str(input_path))
        if not np.allclose(returned_y, ambiguity["conditional_median"], atol=1e-6):
            raise RuntimeError(f"target mismatch for {ckpt}")
        prediction = prediction.astype(np.float64)
        raw_errors = np.abs(prediction[test, None] - fidelity_grid[test])
        rows.append(
            {
                "checkpoint": Path(ckpt).stem,
                "calibration": "raw",
                "repeat": -1,
                "n_calib": 0,
                "n_test_inputs": len(test),
                "n_eta": args.n_eta,
                "mae_grid": float(raw_errors.mean()),
                "mean_worst_abs": float(raw_errors.max(axis=1).mean()),
                "bayes_mae_lower_bound": bayes_test,
                "excess_mae_over_bayes": float(raw_errors.mean()) - bayes_test,
            }
        )
        for repeat in range(args.calib_repeats):
            eta_rng = np.random.default_rng(args.seed + 1_000_003 * (repeat + 1))
            realised = eta_rng.integers(0, args.n_eta, size=len(cal))
            calibration_labels = fidelity_grid[cal, realised]
            slope, intercept = fit_recalibrate(
                prediction[cal], calibration_labels, mode="linear"
            )
            calibrated = np.clip(slope * prediction[test] + intercept, 0.0, 1.0)
            errors = np.abs(calibrated[:, None] - fidelity_grid[test])
            rows.append(
                {
                    "checkpoint": Path(ckpt).stem,
                    "calibration": "one_realised_eta_per_input",
                    "repeat": repeat,
                    "n_calib": args.n_calib,
                    "n_test_inputs": len(test),
                    "n_eta": args.n_eta,
                    "mae_grid": float(errors.mean()),
                    "mean_worst_abs": float(errors.max(axis=1).mean()),
                    "bayes_mae_lower_bound": bayes_test,
                    "excess_mae_over_bayes": float(errors.mean()) - bayes_test,
                }
            )

    result = pd.DataFrame(rows)
    result_path = prefix.with_name(prefix.name + "_models.csv")
    result.to_csv(result_path, index=False)
    summary = (
        result.groupby("calibration")
        .agg(
            mae_mean=("mae_grid", "mean"),
            mae_std=("mae_grid", "std"),
            worst_mean=("mean_worst_abs", "mean"),
            bayes_mae=("bayes_mae_lower_bound", "mean"),
            excess_mean=("excess_mae_over_bayes", "mean"),
            excess_std=("excess_mae_over_bayes", "std"),
            runs=("mae_grid", "size"),
        )
        .reset_index()
    )
    summary_path = prefix.with_name(prefix.name + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(per_base.describe().to_string(float_format=lambda value: f"{value:.5f}"))
    print("\n", summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"[max marginal delta] {max_marginal_delta:.3e}")
    print(f"[saved] {input_path}")
    print(f"[saved] {per_base_path}")
    print(f"[saved] {result_path}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
