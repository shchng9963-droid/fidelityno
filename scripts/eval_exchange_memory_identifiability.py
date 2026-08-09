"""Audit hidden-memory ambiguity in an exchange-coupled collision family."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from physics.baselines.hybrid import ambiguity_statistics
from physics.channels.collision_exchange_memory import (
    exchange_collision_sequence,
    exchange_fidelity_grid_from_params,
    exchange_sequence_from_params,
)
from physics.composition import exact_sequence_fidelity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-base", type=int, default=1024)
    parser.add_argument("--n-eta", type=int, default=15)
    parser.add_argument("--eta-min", type=float, default=0.85)
    parser.add_argument("--eta-max", type=float, default=0.99)
    parser.add_argument("--lengths", default="8,16,24")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--out-prefix", default="results_mlst/exchange_memory_identifiability"
    )
    args = parser.parse_args()

    lengths = np.array(
        sorted({int(value) for value in args.lengths.split(",") if value.strip()})
    )
    eta_grid = np.linspace(args.eta_min, args.eta_max, args.n_eta)
    if args.n_base < 1 or len(lengths) == 0 or np.any(lengths < 1):
        raise ValueError("invalid number of samples or sequence lengths")
    if not 0.0 <= args.eta_min < args.eta_max <= 1.0:
        raise ValueError("eta range must lie in [0, 1]")

    rng = np.random.default_rng(args.seed)
    records = []
    max_marginal_delta = 0.0
    max_eta_zero_delta = 0.0
    for base_index in range(args.n_base):
        length = int(rng.choice(lengths))
        base = exchange_collision_sequence(length, eta=0.0, rng=rng)
        exact_marginal = float(exact_sequence_fidelity(base.marginals))
        grid = exchange_fidelity_grid_from_params(base.params, eta_grid)
        ambiguity = ambiguity_statistics(grid[None, :])

        # eta is absent from every marginal. Replaying the same physical
        # parameters at two eta values must leave the model input unchanged.
        if base_index < min(args.n_base, 64):
            low = exchange_sequence_from_params(base.params, eta=float(eta_grid[0]))
            high = exchange_sequence_from_params(base.params, eta=float(eta_grid[-1]))
            for left, right in zip(low.marginals, high.marginals):
                max_marginal_delta = max(
                    max_marginal_delta,
                    float(np.max(np.abs(left.choi - right.choi))),
                )
            max_eta_zero_delta = max(
                max_eta_zero_delta, abs(base.true_F_e - exact_marginal)
            )

        records.append(
            {
                "base_index": base_index,
                "length": length,
                "fidelity_min": float(ambiguity["lower"][0]),
                "fidelity_max": float(ambiguity["upper"][0]),
                "ambiguity_diameter": float(ambiguity["diameter"][0]),
                "minimax_abs_lower_bound": float(
                    ambiguity["minimax_abs_lower_bound"][0]
                ),
                "bayes_mae_lower_bound": float(
                    ambiguity["uniform_grid_bayes_mae"][0]
                ),
                "exact_marginal_fidelity": exact_marginal,
                "exact_composition_grid_mae": float(np.abs(grid - exact_marginal).mean()),
                "fraction_grid_error_above_0p05": float(
                    np.mean(np.abs(grid - exact_marginal) > 0.05)
                ),
            }
        )

    if max_marginal_delta > 1e-12:
        raise RuntimeError(f"marginals changed across eta: {max_marginal_delta}")
    if max_eta_zero_delta > 1e-9:
        raise RuntimeError(f"eta=0 did not recover composition: {max_eta_zero_delta}")

    result = pd.DataFrame(records)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    detail_path = prefix.with_name(prefix.name + "_per_base.csv")
    result.to_csv(detail_path, index=False)

    summary = (
        result.assign(group="all")
        .groupby("group")
        .agg(
            n_base=("base_index", "size"),
            ambiguity_diameter_mean=("ambiguity_diameter", "mean"),
            ambiguity_diameter_median=("ambiguity_diameter", "median"),
            ambiguity_diameter_q90=("ambiguity_diameter", lambda x: x.quantile(0.9)),
            minimax_abs_lower_bound_mean=("minimax_abs_lower_bound", "mean"),
            bayes_mae_lower_bound_mean=("bayes_mae_lower_bound", "mean"),
            exact_composition_grid_mae_mean=("exact_composition_grid_mae", "mean"),
            fraction_grid_error_above_0p05_mean=(
                "fraction_grid_error_above_0p05",
                "mean",
            ),
        )
        .reset_index()
    )
    by_length = (
        result.groupby("length")
        .agg(
            n_base=("base_index", "size"),
            ambiguity_diameter_mean=("ambiguity_diameter", "mean"),
            minimax_abs_lower_bound_mean=("minimax_abs_lower_bound", "mean"),
            bayes_mae_lower_bound_mean=("bayes_mae_lower_bound", "mean"),
            exact_composition_grid_mae_mean=("exact_composition_grid_mae", "mean"),
        )
        .reset_index()
    )
    summary_path = prefix.with_name(prefix.name + "_summary.csv")
    length_path = prefix.with_name(prefix.name + "_by_length.csv")
    summary.to_csv(summary_path, index=False)
    by_length.to_csv(length_path, index=False)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n" + by_length.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"[max marginal delta] {max_marginal_delta:.3e}")
    print(f"[max eta=0 composition delta] {max_eta_zero_delta:.3e}")
    print(f"[saved] {detail_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {length_path}")


if __name__ == "__main__":
    main()
