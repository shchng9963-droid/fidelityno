"""Evaluate whether a zero-shot surrogate preserves OOD fidelity rankings."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from scripts.eval_recalibrated import predict_means


def set_overlap_score(prediction: np.ndarray, target: np.ndarray, fraction: float, high: bool) -> float:
    count = max(1, int(round(fraction * len(target))))
    if high:
        predicted = np.argpartition(prediction, -count)[-count:]
        actual = np.argpartition(target, -count)[-count:]
    else:
        predicted = np.argpartition(prediction, count - 1)[:count]
        actual = np.argpartition(target, count - 1)[:count]
    return float(len(np.intersect1d(predicted, actual, assume_unique=False)) / count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--fractions", default="0.05,0.10,0.20")
    parser.add_argument("--out", default="results_mlst/ranking_utility.csv")
    args = parser.parse_args()

    fractions = [float(value) for value in args.fractions.split(",") if value.strip()]
    if any(not 0.0 < value < 0.5 for value in fractions):
        raise ValueError("ranking fractions must lie in (0, 0.5)")

    rows = []
    for checkpoint in args.ckpts:
        prediction, target = predict_means(checkpoint, args.data)
        row = {
            "checkpoint": Path(checkpoint).stem,
            "n": len(target),
            "pearson_r": float(pearsonr(prediction, target).statistic),
            "spearman_rho": float(spearmanr(prediction, target).statistic),
            "kendall_tau": float(kendalltau(prediction, target).statistic),
        }
        for fraction in fractions:
            suffix = f"{int(round(100 * fraction))}pct"
            row[f"lowest_{suffix}_overlap"] = set_overlap_score(
                prediction, target, fraction, high=False
            )
            row[f"highest_{suffix}_overlap"] = set_overlap_score(
                prediction, target, fraction, high=True
            )
        rows.append(row)

    result = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    numeric = [column for column in result.columns if column not in {"checkpoint", "n"}]
    summary = pd.DataFrame(
        {
            "metric": numeric,
            "mean": [result[column].mean() for column in numeric],
            "std_across_checkpoints": [result[column].std(ddof=1) for column in numeric],
        }
    )
    summary_path = out.with_name(out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n" + summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"[saved] {out}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
