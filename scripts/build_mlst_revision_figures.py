"""Build the revised MLST cost--accuracy and calibration figures."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "neural": "#1769aa",
    "summary": "#7b1fa2",
    "exact": "#00897b",
    "product": "#43a047",
    "constant": "#757575",
    "dfe": "#d84315",
}


def aggregate_label_budget(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["family"] = work["model"]
    work.loc[work["model"].str.match(r"bidir_seed\d+_affine"), "family"] = "BiDir + affine"
    names = {
        "summary_ridge": "summary ridge",
        "exact_marginals_affine": "exact marginals + affine",
        "product_affine": "product + affine",
        "constant_cal_median": "calibration median",
    }
    work["family"] = work["family"].replace(names)
    keep = ["BiDir + affine", *names.values()]
    return (
        work[work["family"].isin(keep)]
        .groupby(["family", "n_calib"])
        .agg(mae=("mae_F_e", "mean"), std=("mae_F_e", "std"))
        .reset_index()
    )


def plot_calibration(ax: plt.Axes, agg: pd.DataFrame) -> None:
    styles = {
        "BiDir + affine": (COLORS["neural"], "o", "-"),
        "summary ridge": (COLORS["summary"], "s", "-"),
        "exact marginals + affine": (COLORS["exact"], "^", "--"),
        "product + affine": (COLORS["product"], "v", "--"),
        "calibration median": (COLORS["constant"], "D", ":"),
    }
    for name, (color, marker, line) in styles.items():
        part = agg[agg["family"] == name].sort_values("n_calib")
        ax.plot(part["n_calib"], part["mae"], line, marker=marker, color=color, label=name)
        if name == "BiDir + affine":
            std = part["std"].fillna(0.0)
            ax.fill_between(part["n_calib"], part["mae"] - std, part["mae"] + std,
                            color=color, alpha=0.16, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"labelled OOD samples $N_{cal}$")
    ax.set_ylabel(r"family-shift MAE in $F_e$")
    ax.grid(alpha=0.25)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_mlst")
    ap.add_argument("--outdir", default="results_mlst/figures")
    args = ap.parse_args()
    root = Path(args.results)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    exact = pd.read_csv(root / "exact_composition_optimized.csv")
    labels = pd.read_csv(root / "collision_family_ood_label_budget.csv")
    dfe = pd.read_csv(root / "dfe_family_ood_low_shot.csv")
    agg = aggregate_label_budget(labels)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.55))
    ax = axes[0]
    is_collision = exact["path"].str.contains("collision")
    for collision, marker, color, label in [
        (False, "o", COLORS["exact"], "Markovian (exact target)"),
        (True, "s", COLORS["dfe"], "collision marginals"),
    ]:
        part = exact[is_collision == collision]
        ax.scatter(part["latency_ms_per_seq_median"], part["mae_F_e"], s=55,
                   marker=marker, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("CPU latency (ms/sequence)")
    ax.set_ylabel(r"MAE in $F_e$")
    ax.set_title("(a) Exact composition")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    plot_calibration(axes[1], agg)
    axes[1].set_title("(b) Same OOD-label budget")
    axes[1].legend(frameon=False, fontsize=7.5)

    ax = axes[2]
    ax.plot(dfe["quantum_shots_per_seq"], dfe["mae_F_e"], "-o", color=COLORS["dfe"],
            label="stratified DFE")
    neural64 = agg[(agg["family"] == "BiDir + affine") & (agg["n_calib"] == 64)]["mae"].iloc[0]
    ax.axhline(neural64, color=COLORS["neural"], linestyle="--",
               label=rf"BiDir+cal ($N_{{cal}}=64$): {neural64:.3f}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("quantum shots per query")
    ax.set_ylabel(r"family-shift MAE in $F_e$")
    ax.set_title("(c) Measurement crossover")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "central_sample_complexity.pdf", bbox_inches="tight")
    fig.savefig(out / "central_sample_complexity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.7, 3.8))
    plot_calibration(ax, agg)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "calibration_sweep.pdf", bbox_inches="tight")
    fig.savefig(out / "calibration_sweep.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.7, 3.8))
    ax.plot(dfe["quantum_shots_per_seq"], dfe["mae_F_e"], "-o", color=COLORS["dfe"],
            label="stratified DFE")
    ax.axhline(neural64, color=COLORS["neural"], linestyle="--", label="BiDir + exact-label cal")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("quantum shots per query")
    ax.set_ylabel(r"family-shift MAE in $F_e$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "dfe_sample_complexity.pdf", bbox_inches="tight")
    fig.savefig(out / "dfe_sample_complexity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    q = np.arange(1, 513)
    fig, ax = plt.subplots(figsize=(5.7, 3.8))
    ax.plot(q, 32 * q, color=COLORS["dfe"], label="DFE 32 shots/query (MAE 0.092)")
    ax.plot(q, 64 * q, color="#ef6c00", label="DFE 64 shots/query (MAE 0.065)")
    ax.axhline(64 * 64, color=COLORS["neural"], linestyle="--",
               label="BiDir calibration: 64 labels $\\times$ 64 shots")
    ax.axvline(128, color="#9e9e9e", linestyle=":", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("deployment queries")
    ax.set_ylabel("cumulative quantum shots")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "break_even_queries.pdf", bbox_inches="tight")
    fig.savefig(out / "break_even_queries.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] revised figures in {out}")


if __name__ == "__main__":
    main()
