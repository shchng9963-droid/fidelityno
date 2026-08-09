"""Evaluate two-stage, query-adaptive DFE shot allocation.

Every query first receives a small stratified DFE pilot.  A fixed batch budget
is then split between two final shot levels.  The high level is assigned to
queries with the largest prior-pilot disagreement, predictive interval width,
or their rank-average.  A random allocation control uses the same shot totals.
Fusion weights are fitted only on the held-out calibration set.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from physics.baselines.hybrid import (
    allocate_two_level_budget,
    apply_budgeted_convex_fusion,
    apply_convex_fusion,
    batch_pauli_expectations_from_choi,
    complete_identity_dfe,
    fit_budgeted_convex_fusion,
    fit_convex_fusion,
    sample_identity_dfe,
    sample_identity_dfe_pilot,
)
from scripts.eval_label_budget_baselines import metrics
from scripts.eval_recalibrated import fit_recalibrate, predict_quantiles


def fractional_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.lexsort((np.arange(len(values)), values))
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(len(values) - 1, 1)


def metric_row(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return metrics(np.clip(prediction, 0.0, 1.0), target)


def add_result(
    rows: list[dict],
    errors: dict[str, list[np.ndarray]],
    *,
    key: str,
    method: str,
    policy: str,
    checkpoint: str,
    repeat: int,
    average_shots: int,
    low_shots: int,
    high_shots: int,
    offline_shots: int,
    prediction: np.ndarray,
    target: np.ndarray,
    weights: dict[int, float] | None = None,
    record_row: bool = True,
) -> None:
    absolute_error = np.abs(np.clip(prediction, 0.0, 1.0) - target)
    errors[key].append(absolute_error)
    if not record_row:
        return
    row = {
        "method": method,
        "policy": policy,
        "checkpoint": checkpoint,
        "repeat": repeat,
        "average_query_shots": average_shots,
        "low_shots": low_shots,
        "high_shots": high_shots,
        "offline_quantum_shots": offline_shots,
        "weight_low": np.nan,
        "weight_high": np.nan,
    }
    if weights:
        row["weight_low"] = weights.get(low_shots, np.nan)
        row["weight_high"] = weights.get(high_shots, np.nan)
    row.update(metric_row(prediction, target))
    rows.append(row)


def bootstrap_run_delta(
    left: list[np.ndarray],
    right: list[np.ndarray],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    n_checkpoints: int,
    n_repeats: int,
) -> tuple[float, float, float, int]:
    if len(left) != len(right) or not left:
        raise ValueError("paired error lists must have the same non-zero length")
    if len(left) != n_checkpoints * n_repeats:
        raise ValueError("paired lists do not match the crossed-design dimensions")
    run_delta = np.array(
        [float(np.mean(a - b)) for a, b in zip(left, right)], dtype=np.float64
    ).reshape(n_checkpoints, n_repeats)
    checkpoint_draws = rng.integers(
        0, n_checkpoints, size=(n_bootstrap, n_checkpoints)
    )
    repeat_draws = rng.integers(0, n_repeats, size=(n_bootstrap, n_repeats))
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        boot[index] = run_delta[
            np.ix_(checkpoint_draws[index], repeat_draws[index])
        ].mean()
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(run_delta.mean()), float(low), float(high), run_delta.size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--average-budgets", default="32,64")
    ap.add_argument("--pilot-shots", type=int, default=8)
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=4032)
    ap.add_argument("--label-shots", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=20260806)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument(
        "--out", default="results_mlst/adaptive_measurement_allocation.csv"
    )
    args = ap.parse_args()

    average_budgets = [
        int(value) for value in args.average_budgets.split(",") if value.strip()
    ]
    if not average_budgets or any(value % 8 for value in average_budgets):
        raise ValueError("average budgets must be positive multiples of eight")
    if args.pilot_shots < 4 or args.pilot_shots % 4:
        raise ValueError("pilot shots must be a positive multiple of four")

    raw = np.load(args.data, allow_pickle=True)
    y = raw["y"].astype(np.float64)
    if args.n_calib < 8 or args.n_calib + args.n_test > len(y):
        raise ValueError("invalid calibration/test sizes")
    permutation = np.random.default_rng(args.split_seed).permutation(len(y))
    cal = permutation[: args.n_calib]
    test = permutation[args.n_calib : args.n_calib + args.n_test]

    choi = raw["true_choi_real"].astype(np.float64)
    choi = choi + 1j * raw["true_choi_imag"].astype(np.float64)
    expectations = batch_pauli_expectations_from_choi(choi)
    max_label_delta = float(np.max(np.abs(expectations.mean(axis=1) - y)))
    if max_label_delta > 5e-6:
        raise RuntimeError(f"Pauli/label fidelity mismatch: {max_label_delta}")

    model_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ckpt in args.ckpts:
        quantiles, returned_y = predict_quantiles(ckpt, args.data)
        if not np.allclose(returned_y, y, atol=1e-6):
            raise RuntimeError(f"ground-truth mismatch for {ckpt}")
        name = Path(ckpt).stem
        model_outputs[name] = (
            quantiles.mean(axis=1).astype(np.float64),
            (quantiles[:, -1] - quantiles[:, 0]).astype(np.float64),
        )

    labels_by_repeat: dict[int, np.ndarray] = {}
    pilot_by_repeat: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    fixed_by_repeat: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for repeat in range(args.repeats):
        label_rng = np.random.default_rng(args.split_seed + 10_000_019 * (repeat + 1))
        labels_by_repeat[repeat], _ = sample_identity_dfe(
            expectations[cal], args.label_shots, label_rng
        )
        cal_pilot_rng = np.random.default_rng(
            args.split_seed + 20_000_033 * (repeat + 1)
        )
        test_pilot_rng = np.random.default_rng(
            args.split_seed + 30_000_077 * (repeat + 1)
        )
        cal_plus, cal_pilot = sample_identity_dfe_pilot(
            expectations[cal], args.pilot_shots, cal_pilot_rng
        )
        test_plus, test_pilot = sample_identity_dfe_pilot(
            expectations[test], args.pilot_shots, test_pilot_rng
        )
        pilot_by_repeat[repeat] = (cal_plus, cal_pilot, test_plus, test_pilot)
        for budget in sorted(set(average_budgets + [2 * b for b in average_budgets])):
            cal_rng = np.random.default_rng(
                args.split_seed + 40_000_087 * (repeat + 1) + budget
            )
            test_rng = np.random.default_rng(
                args.split_seed + 50_000_099 * (repeat + 1) + budget
            )
            cal_dfe, cal_sigma = sample_identity_dfe(
                expectations[cal], budget, cal_rng
            )
            test_dfe, test_sigma = sample_identity_dfe(
                expectations[test], budget, test_rng
            )
            fixed_by_repeat[(repeat, budget)] = (
                cal_dfe,
                cal_sigma,
                test_dfe,
                test_sigma,
            )

    rows: list[dict] = []
    errors: dict[str, list[np.ndarray]] = defaultdict(list)
    policies = ("random", "width", "disagreement", "combined")
    label_cost = args.n_calib * args.label_shots
    first_checkpoint = next(iter(model_outputs))

    for checkpoint, (raw_prior, raw_width) in model_outputs.items():
        for repeat in range(args.repeats):
            finite_labels = labels_by_repeat[repeat]
            slope, intercept = fit_recalibrate(
                raw_prior[cal], finite_labels, mode="linear"
            )
            prior = np.clip(slope * raw_prior + intercept, 0.0, 1.0)
            width = np.maximum(abs(slope) * raw_width, 1e-8)
            cal_plus, cal_pilot, test_plus, test_pilot = pilot_by_repeat[repeat]

            for average in average_budgets:
                low = average // 2
                high = 2 * average - low
                if low < args.pilot_shots:
                    raise ValueError("low budget must include the pilot")

                cal_dfe, _, test_dfe, _ = fixed_by_repeat[(repeat, average)]
                fixed_weight = fit_convex_fusion(
                    prior[cal], cal_dfe, finite_labels
                )
                fixed_hybrid = np.clip(
                    apply_convex_fusion(prior[test], test_dfe, fixed_weight),
                    0.0,
                    1.0,
                )
                fixed_weights = {average: fixed_weight}
                add_result(
                    rows,
                    errors,
                    key=f"fixed_hybrid|fixed|{average}",
                    method="fixed_hybrid",
                    policy="fixed",
                    checkpoint=checkpoint,
                    repeat=repeat,
                    average_shots=average,
                    low_shots=average,
                    high_shots=average,
                    offline_shots=label_cost + args.n_calib * average,
                    prediction=fixed_hybrid,
                    target=y[test],
                    weights=fixed_weights,
                )
                add_result(
                    rows,
                    errors,
                    key=f"fixed_dfe|fixed|{average}",
                    method="fixed_dfe",
                    policy="fixed",
                    checkpoint=checkpoint,
                    repeat=repeat,
                    average_shots=average,
                    low_shots=average,
                    high_shots=average,
                    offline_shots=0,
                    prediction=test_dfe,
                    target=y[test],
                    record_row=checkpoint == first_checkpoint,
                )

                _, _, test_dfe_double, _ = fixed_by_repeat[(repeat, 2 * average)]
                # A doubled budget can also be one of the requested average
                # budgets. In that case its ordinary fixed-DFE row is added
                # by the corresponding loop iteration and must not be counted
                # twice in paired comparisons.
                if 2 * average not in average_budgets:
                    add_result(
                        rows,
                        errors,
                        key=f"fixed_dfe|fixed|{2 * average}",
                        method="fixed_dfe",
                        policy="fixed",
                        checkpoint=checkpoint,
                        repeat=repeat,
                        average_shots=2 * average,
                        low_shots=2 * average,
                        high_shots=2 * average,
                        offline_shots=0,
                        prediction=test_dfe_double,
                        target=y[test],
                        record_row=checkpoint == first_checkpoint,
                    )

                random_rng = np.random.default_rng(
                    args.split_seed
                    + 60_000_119 * (repeat + 1)
                    + 101 * average
                    + sum(ord(char) for char in checkpoint)
                )
                score_pairs = {
                    "random": (random_rng.random(len(cal)), random_rng.random(len(test))),
                    "width": (width[cal], width[test]),
                    "disagreement": (
                        np.abs(cal_pilot - prior[cal]),
                        np.abs(test_pilot - prior[test]),
                    ),
                    "combined": (
                        fractional_rank(width[cal])
                        + fractional_rank(np.abs(cal_pilot - prior[cal])),
                        fractional_rank(width[test])
                        + fractional_rank(np.abs(test_pilot - prior[test])),
                    ),
                }

                for policy in policies:
                    cal_score, test_score = score_pairs[policy]
                    cal_budget = allocate_two_level_budget(cal_score, low, high)
                    test_budget = allocate_two_level_budget(test_score, low, high)
                    cal_extra_rng = np.random.default_rng(
                        args.split_seed
                        + 70_000_133 * (repeat + 1)
                        + 1009 * average
                        + 37 * policies.index(policy)
                        + sum(ord(char) for char in checkpoint)
                    )
                    test_extra_rng = np.random.default_rng(
                        args.split_seed
                        + 80_000_153 * (repeat + 1)
                        + 1013 * average
                        + 41 * policies.index(policy)
                        + sum(ord(char) for char in checkpoint)
                    )
                    cal_final, _ = complete_identity_dfe(
                        expectations[cal],
                        cal_plus,
                        args.pilot_shots,
                        cal_budget,
                        cal_extra_rng,
                    )
                    test_final, _ = complete_identity_dfe(
                        expectations[test],
                        test_plus,
                        args.pilot_shots,
                        test_budget,
                        test_extra_rng,
                    )
                    weights = fit_budgeted_convex_fusion(
                        prior[cal], cal_final, finite_labels, cal_budget
                    )
                    adaptive_hybrid = np.clip(
                        apply_budgeted_convex_fusion(
                            prior[test], test_final, test_budget, weights
                        ),
                        0.0,
                        1.0,
                    )
                    offline = label_cost + int(cal_budget.sum())
                    add_result(
                        rows,
                        errors,
                        key=f"adaptive_hybrid|{policy}|{average}",
                        method="adaptive_hybrid",
                        policy=policy,
                        checkpoint=checkpoint,
                        repeat=repeat,
                        average_shots=int(round(test_budget.mean())),
                        low_shots=low,
                        high_shots=high,
                        offline_shots=offline,
                        prediction=adaptive_hybrid,
                        target=y[test],
                        weights=weights,
                    )
                    add_result(
                        rows,
                        errors,
                        key=f"adaptive_dfe|{policy}|{average}",
                        method="adaptive_dfe",
                        policy=policy,
                        checkpoint=checkpoint,
                        repeat=repeat,
                        average_shots=int(round(test_budget.mean())),
                        low_shots=low,
                        high_shots=high,
                        offline_shots=0,
                        prediction=test_final,
                        target=y[test],
                    )

    result = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    summary = (
        result.groupby(
            ["method", "policy", "average_query_shots", "low_shots", "high_shots"],
            dropna=False,
        )
        .agg(
            mae_mean=("mae_F_e", "mean"),
            mae_std=("mae_F_e", "std"),
            rmse_mean=("rmse_F_e", "mean"),
            weight_low_mean=("weight_low", "mean"),
            weight_high_mean=("weight_high", "mean"),
            offline_shots=("offline_quantum_shots", "max"),
            runs=("mae_F_e", "size"),
        )
        .reset_index()
    )
    summary_path = out.with_name(out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    comparison_rows = []
    comparison_rng = np.random.default_rng(args.split_seed + 91_000_177)
    for average in average_budgets:
        fixed_key = f"fixed_hybrid|fixed|{average}"
        dfe_double_key = f"fixed_dfe|fixed|{2 * average}"
        comparisons = [
            ("fixed_hybrid_vs_double_dfe", fixed_key, dfe_double_key),
            (
                "adaptive_combined_vs_fixed_hybrid",
                f"adaptive_hybrid|combined|{average}",
                fixed_key,
            ),
            (
                "adaptive_disagreement_vs_fixed_hybrid",
                f"adaptive_hybrid|disagreement|{average}",
                fixed_key,
            ),
            (
                "adaptive_width_vs_fixed_hybrid",
                f"adaptive_hybrid|width|{average}",
                fixed_key,
            ),
            (
                "adaptive_random_vs_fixed_hybrid",
                f"adaptive_hybrid|random|{average}",
                fixed_key,
            ),
            (
                "adaptive_combined_vs_double_dfe",
                f"adaptive_hybrid|combined|{average}",
                dfe_double_key,
            ),
        ]
        for name, left_key, right_key in comparisons:
            delta, low_ci, high_ci, runs = bootstrap_run_delta(
                errors[left_key],
                errors[right_key],
                rng=comparison_rng,
                n_bootstrap=args.bootstrap,
                n_checkpoints=len(model_outputs),
                n_repeats=args.repeats,
            )
            comparison_rows.append(
                {
                    "comparison": name,
                    "average_query_shots": average,
                    "left_key": left_key,
                    "right_key": right_key,
                    "delta_mae_left_minus_right": delta,
                    "ci95_low": low_ci,
                    "ci95_high": high_ci,
                    "paired_runs": runs,
                    "checkpoint_clusters": len(model_outputs),
                    "measurement_repeat_clusters": args.repeats,
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    comparison_path = out.with_name(out.stem + "_paired_ci.csv")
    comparison.to_csv(comparison_path, index=False)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n", comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"[max Pauli-label delta] {max_label_delta:.3e}")
    print(f"[saved] {out}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {comparison_path}")


if __name__ == "__main__":
    main()
