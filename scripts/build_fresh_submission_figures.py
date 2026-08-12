#!/usr/bin/env python3
"""Rebuild all MLST submission figures with one clean, overlap-safe style."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_mlst"
OUT = ROOT / "manuscript" / "mlst" / "figures"

# FRESH: restrained colour-blind-safe hues, open axes, light guides, and
# consistent typography.  The palette is also distinguishable in greyscale
# through marker and line-style changes.
NAVY = "#183B56"
BLUE = "#2A6FBB"
TEAL = "#168C80"
CORAL = "#E56B4A"
AMBER = "#D99B2B"
PURPLE = "#7656A8"
SLATE = "#64748B"
LIGHT_BLUE = "#9BC4E2"
GRID = "#DCE5EA"
INK = "#25313B"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.labelsize": 8.2,
    "axes.titlesize": 8.7,
    "axes.titleweight": "semibold",
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.75,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "legend.fontsize": 6.9,
    "legend.frameon": False,
    "lines.linewidth": 1.55,
    "lines.markersize": 4.2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def finish_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.85)
    ax.set_axisbelow(True)


def panel_title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", pad=7)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "build_fresh_submission_figures.py"}
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", metadata=metadata)
    plt.close(fig)
    print(f"[saved] {stem}.pdf")


def label_budget_aggregate() -> pd.DataFrame:
    data = pd.read_csv(RESULTS / "collision_family_ood_label_budget.csv")
    data["family"] = data["model"]
    data.loc[data["model"].str.match(r"bidir_seed\d+_affine"), "family"] = "BiDir + affine"
    data["family"] = data["family"].replace({
        "summary_ridge": "summary ridge",
        "exact_marginals_affine": "exact marginals + affine",
        "product_affine": "product + affine",
        "constant_cal_median": "calibration median",
    })
    keep = ["BiDir + affine", "summary ridge", "exact marginals + affine",
            "product + affine", "calibration median"]
    return (data[data["family"].isin(keep)]
            .groupby(["family", "n_calib"])
            .agg(mae=("mae_F_e", "mean"), std=("mae_F_e", "std"))
            .reset_index())


CAL_STYLES = {
    "BiDir + affine": (BLUE, "o", "-"),
    "summary ridge": (PURPLE, "s", "-"),
    "exact marginals + affine": (TEAL, "^", "--"),
    "product + affine": (AMBER, "v", "--"),
    "calibration median": (SLATE, "D", ":"),
}


def plot_label_budget(ax: plt.Axes, agg: pd.DataFrame) -> None:
    for name, (colour, marker, line) in CAL_STYLES.items():
        part = agg[agg["family"] == name].sort_values("n_calib")
        ax.plot(part["n_calib"], part["mae"], linestyle=line, marker=marker,
                color=colour, label=name, markeredgecolor="white", markeredgewidth=0.35)
        if name == "BiDir + affine":
            std = part["std"].fillna(0.0)
            ax.fill_between(part["n_calib"], part["mae"] - std,
                            part["mae"] + std, color=colour, alpha=0.13, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64, 128, 256],
                  labels=[r"$2^3$", r"$2^4$", r"$2^5$", r"$2^6$", r"$2^7$", r"$2^8$"])
    ax.set_xlabel(r"labelled OOD samples $N_{\mathrm{cal}}$")
    ax.set_ylabel(r"family-shift MAE in $F_{\mathrm{e}}$")
    finish_axis(ax)


def central_sample_complexity() -> None:
    exact = pd.read_csv(RESULTS / "exact_composition_optimized.csv")
    dfe = pd.read_csv(RESULTS / "dfe_family_ood_low_shot.csv")
    agg = label_budget_aggregate()
    neural64 = float(agg[(agg["family"] == "BiDir + affine") &
                         (agg["n_calib"] == 64)]["mae"].iloc[0])

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.75))
    is_collision = exact["path"].str.contains("collision")
    for flag, marker, colour, label in [
        (False, "o", TEAL, "Markovian, exact target"),
        (True, "s", CORAL, "collision marginals"),
    ]:
        part = exact[is_collision == flag]
        axes[0].scatter(part["latency_ms_per_seq_median"], part["mae_F_e"],
                        s=31, marker=marker, color=colour, edgecolor="white",
                        linewidth=0.5, label=label, zorder=3)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("CPU latency (ms/sequence)")
    axes[0].set_ylabel(r"MAE in $F_{\mathrm{e}}$")
    panel_title(axes[0], "(a) Exact composition")
    finish_axis(axes[0], grid_axis="both")

    plot_label_budget(axes[1], agg)
    panel_title(axes[1], "(b) Matched label budget")

    axes[2].plot(dfe["quantum_shots_per_seq"], dfe["mae_F_e"], "-o",
                 color=CORAL, label="stratified DFE", markeredgecolor="white",
                 markeredgewidth=0.35)
    axes[2].axhline(neural64, color=BLUE, linestyle="--", linewidth=1.4,
                    label=rf"BiDir + cal: {neural64:.3f}")
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("quantum shots per query")
    axes[2].set_ylabel(r"family-shift MAE in $F_{\mathrm{e}}$")
    panel_title(axes[2], "(c) Measurement crossover")
    finish_axis(axes[2], grid_axis="both")

    for ax, cols in zip(axes, [1, 2, 1]):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=cols,
                  handlelength=1.8, columnspacing=0.9)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.31, wspace=0.42)
    save(fig, "central_sample_complexity")


def identifiability_hybrid() -> None:
    data = np.load(RESULTS / "collision_ood_identifiability_inputs.npz")
    eta, grid = data["eta_grid"], data["fidelity_grid"]
    diameter = np.ptp(grid, axis=1)
    med = np.median(grid, axis=1, keepdims=True)
    bayes = np.mean(np.abs(grid - med), axis=1)
    table = pd.read_csv(RESULTS / "measurement_conditioned_hybrid_independent_rng_summary.csv")
    hybrid = table[(table.method == "measurement_conditioned_hybrid") &
                   (table.prior == "bidir_affine") &
                   (table.calibration == "finite_64shot_labels")].sort_values("per_query_shots")
    dfe = table[table.method == "stratified_dfe"].sort_values("per_query_shots")
    prior = table[(table.method == "marginal_only_prior") &
                  (table.prior == "bidir_affine") &
                  (table.calibration == "finite_64shot_labels")].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.05),
                             gridspec_kw={"width_ratios": [1.02, 0.95, 1.13]})
    order = np.argsort(diameter)
    chosen = order[np.linspace(0, len(order) - 1, 15).astype(int)]
    for idx in chosen:
        axes[0].plot(eta, grid[idx], color=LIGHT_BLUE, alpha=0.62, linewidth=0.75)
    axes[0].plot(eta, np.median(grid, axis=0), color=NAVY, linewidth=2.0,
                 label="pointwise median")
    axes[0].set(xlabel=r"bath retention $\eta$", ylabel=r"$F_{\mathrm{e}}$",
                xlim=(eta.min(), eta.max()), ylim=(0, 1))
    panel_title(axes[0], "(a) Fixed input, hidden memory")
    axes[0].legend(loc="lower left", handlelength=2.0)
    finish_axis(axes[0])

    bins = np.linspace(0, max(0.62, diameter.max()), 24)
    axes[1].hist(diameter, bins=bins, color=BLUE, alpha=0.86,
                 edgecolor="white", linewidth=0.35)
    axes[1].axvline(diameter.mean(), color=CORAL, linewidth=1.65)
    axes[1].set(xlabel=r"ambiguity diameter $\Delta_{\phi}(x)$", ylabel="input groups")
    panel_title(axes[1], "(b) Ambiguity across inputs")
    finish_axis(axes[1])

    axes[2].errorbar(dfe.per_query_shots, dfe.mae_mean, yerr=dfe.mae_std,
                     color=CORAL, marker="o", capsize=2, label="stratified DFE",
                     markeredgecolor="white", markeredgewidth=0.35)
    axes[2].errorbar(hybrid.per_query_shots, hybrid.mae_mean, yerr=hybrid.mae_std,
                     color=TEAL, marker="s", capsize=2,
                     label="measurement-conditioned hybrid",
                     markeredgecolor="white", markeredgewidth=0.35)
    axes[2].axhline(prior.mae_mean, color=PURPLE, linestyle="-.", linewidth=1.35,
                    label="marginal neural prior")
    axes[2].axhline(bayes.mean(), color=SLATE, linestyle="--", linewidth=1.25,
                    label="marginal-only Bayes floor")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks([4, 8, 16, 32, 64, 128], labels=[4, 8, 16, 32, 64, 128])
    axes[2].set(xlabel="projective shots per query", ylabel="MAE", ylim=(0.035, 0.295))
    panel_title(axes[2], "(c) Same-query measurement")
    finish_axis(axes[2])

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.78, 0.005),
               ncol=2, columnspacing=0.9, handlelength=2.0)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.25, wspace=0.43)
    save(fig, "identifiability_hybrid")


def break_even() -> None:
    q = np.arange(0, 701)
    fig, ax = plt.subplots(figsize=(4.9, 3.05))
    lines = [
        (64 * q, CORAL, "-", "DFE, 64 shots/query (MAE 0.0641)"),
        (6144 + 32 * q, TEAL, "-", "hybrid, 32 shots/query (MAE 0.0657)"),
        (96 * q, AMBER, "--", "DFE, 96 shots/query (MAE 0.0525)"),
        (8192 + 64 * q, BLUE, "--", "hybrid, 64 shots/query (MAE 0.0539)"),
    ]
    for y, colour, style, label in lines:
        ax.plot(q, y, color=colour, linestyle=style, label=label)
    for x in [192, 256]:
        ax.axvline(x, color=SLATE, linewidth=0.9, linestyle=":")
        ax.text(x, 2100, str(x), ha="center", va="bottom", color=SLATE, fontsize=7.0)
    ax.set(xlabel="deployment queries", ylabel="cumulative projective shots",
           xlim=(0, 700), ylim=(0, 68000))
    finish_axis(ax, grid_axis="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              columnspacing=1.0, handlelength=2.2)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.17, top=0.78)
    save(fig, "break_even_queries")


def eta_sweep() -> None:
    data = pd.read_csv(RESULTS / "eta_sweep_aggregate.csv")
    styles = {
        "product_bound": (AMBER, "^", "product approximation"),
        "mc_1000": (PURPLE, "D", "MC, $K=1000$"),
        "fidelityno_raw": (CORAL, "o", "FidelityFormer, raw"),
        "fidelityno_cal": (TEAL, "s", "FidelityFormer + cal"),
    }
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for model, (colour, marker, label) in styles.items():
        part = data[data.model == model].sort_values("eta")
        ax.plot(part.eta, part.mae_mean, color=colour, marker=marker, label=label,
                markeredgecolor="white", markeredgewidth=0.35)
        if part.mae_std.notna().any():
            ax.fill_between(part.eta, part.mae_mean - part.mae_std.fillna(0),
                            part.mae_mean + part.mae_std.fillna(0),
                            color=colour, alpha=0.12, linewidth=0)
    ax.set(xlabel=r"bath retention $\eta$", ylabel="MAE", xlim=(-0.02, 1.01))
    finish_axis(ax, grid_axis="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              columnspacing=1.1, handlelength=2.0)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.17, top=0.79)
    save(fig, "eta_sweep")


def calibration_sweep() -> None:
    fig, ax = plt.subplots(figsize=(4.9, 3.45))
    plot_label_budget(ax, label_budget_aggregate())
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              columnspacing=1.0, handlelength=2.1)
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.16, top=0.75)
    save(fig, "calibration_sweep")


def calibration_ece() -> None:
    data = pd.read_csv(RESULTS / "calibration_ece_summary.csv").sort_values("ece_mean", ascending=False)
    labels = {
        "fidelityno": "FidelityFormer", "fidelityno_large": "FidelityFormer-L",
        "bidir": "BiDir Transformer", "mlp": "MLP", "deepsets": "DeepSets",
        "gnn": "GNN", "generic_gnn": "Generic GNN", "product_bound": "Product approximation",
        "fvg_bound": "FvG bound", "mc_10": "MC, $K=10$", "mc_100": "MC, $K=100$",
        "mc_1000": "MC, $K=1000$",
    }
    colours = []
    for model in data.model:
        if model in {"product_bound", "fvg_bound"}:
            colours.append(AMBER)
        elif model.startswith("mc_"):
            colours.append(SLATE)
        elif model in {"fidelityno", "fidelityno_large", "bidir"}:
            colours.append(BLUE)
        else:
            colours.append(TEAL)
    fig, ax = plt.subplots(figsize=(5.25, 3.55))
    y = np.arange(len(data))
    ax.barh(y, data.ece_mean, xerr=data.ece_std, color=colours, alpha=0.9,
            edgecolor="white", linewidth=0.45, error_kw={"ecolor": INK, "elinewidth": 0.7, "capsize": 1.8})
    ax.set_yticks(y, labels=[labels[m] for m in data.model])
    ax.axvline(0.05, color=CORAL, linestyle="--", linewidth=1.15)
    ax.text(0.05, 1.01, "ECE = 0.05", color=CORAL, ha="center", va="bottom",
            transform=ax.get_xaxis_transform(), fontsize=7.0)
    ax.set_xlabel("expected calibration error (ECE)")
    ax.set_xlim(0, 0.53)
    finish_axis(ax, grid_axis="x")
    fig.subplots_adjust(left=0.32, right=0.98, bottom=0.15, top=0.94)
    save(fig, "calibration_ece")


REL_COLOURS = {
    "FidelityNO (5M)": NAVY, "FidelityNO (1M)": LIGHT_BLUE,
    "Generic-GNN": TEAL, "Bidir Trans.": PURPLE,
    "Flat MLP": AMBER, "DeepSets": CORAL,
}
REL_MARKERS = {
    "FidelityNO (5M)": "o", "FidelityNO (1M)": "s", "Generic-GNN": "^",
    "Bidir Trans.": "D", "Flat MLP": "v", "DeepSets": "P",
}
REL_LABELS = {
    "FidelityNO (5M)": "FidelityFormer (5M)",
    "FidelityNO (1M)": "FidelityFormer (1M)",
    "Generic-GNN": "Generic GNN",
    "Bidir Trans.": "BiDir Transformer",
    "Flat MLP": "MLP",
    "DeepSets": "DeepSets",
}


def reliability(split: str, stem: str, title: str) -> None:
    raw = pd.read_csv(RESULTS / "reliability_curves.csv")
    raw = raw[raw.split == split]
    agg = (raw.groupby(["model", "nominal_level"])
           .agg(mean=("empirical_coverage", "mean"), std=("empirical_coverage", "std"))
           .reset_index())
    fig, ax = plt.subplots(figsize=(3.65, 3.35))
    ax.plot([0, 1], [0, 1], color=SLATE, linestyle="--", linewidth=1.0, label="ideal")
    for model in REL_COLOURS:
        part = agg[agg.model == model].sort_values("nominal_level")
        if part.empty:
            continue
        colour = REL_COLOURS[model]
        ax.plot(part.nominal_level, part["mean"], color=colour,
                marker=REL_MARKERS[model], label=REL_LABELS[model],
                markeredgecolor="white", markeredgewidth=0.3)
        ax.fill_between(part.nominal_level, part["mean"] - part["std"].fillna(0),
                        part["mean"] + part["std"].fillna(0),
                        color=colour, alpha=0.10, linewidth=0)
    ax.set(xlabel="nominal quantile level", ylabel="empirical coverage",
           xlim=(0, 1), ylim=(0, 1))
    panel_title(ax, title)
    finish_axis(ax, grid_axis="both")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
              columnspacing=0.8, handlelength=1.8)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.34, top=0.88)
    save(fig, stem)


def main() -> None:
    central_sample_complexity()
    identifiability_hybrid()
    break_even()
    eta_sweep()
    calibration_sweep()
    calibration_ece()
    reliability("ID", "reliability_id", "ID test")
    reliability("Length OOD", "reliability_len48", r"Length OOD, $n=48$")


if __name__ == "__main__":
    main()
