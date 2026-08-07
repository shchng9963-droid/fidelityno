from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_mlst"
FIGURES = ROOT / "manuscript" / "mlst" / "figures"

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 8.2,
    "axes.labelsize": 8.2,
    "axes.titlesize": 8.6,
    "legend.fontsize": 7.1,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BLUE = "#2C6DB2"
ORANGE = "#D97924"
GREEN = "#278B6B"
GREY = "#777777"
PURPLE = "#7A5195"


def load_hybrid():
    table = pd.read_csv(
        RESULTS / "measurement_conditioned_hybrid_independent_rng_summary.csv"
    )
    hybrid = table[
        (table.method == "measurement_conditioned_hybrid")
        & (table.prior == "bidir_affine")
        & (table.calibration == "finite_64shot_labels")
    ].sort_values("per_query_shots")
    dfe = table[table.method == "stratified_dfe"].sort_values("per_query_shots")
    prior = table[
        (table.method == "marginal_only_prior")
        & (table.prior == "bidir_affine")
        & (table.calibration == "finite_64shot_labels")
    ].iloc[0]
    return hybrid, dfe, prior


def identifiability_and_hybrid():
    data = np.load(RESULTS / "collision_ood_identifiability_inputs.npz")
    eta = data["eta_grid"]
    grid = data["fidelity_grid"]
    diam = np.ptp(grid, axis=1)
    conditional_median = np.median(grid, axis=1, keepdims=True)
    bayes_by_input = np.mean(np.abs(grid - conditional_median), axis=1)
    hybrid, dfe, prior = load_hybrid()

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.35))
    ax = axes[0]
    order = np.argsort(diam)
    chosen = order[np.linspace(0, len(order) - 1, 17).astype(int)]
    for idx in chosen:
        ax.plot(eta, grid[idx], color=BLUE, alpha=0.25, lw=0.75)
    ax.plot(eta, np.median(grid, axis=0), color="#173A5E", lw=1.8,
            label="pointwise median")
    ax.set(xlabel=r"bath retention $\eta$", ylabel=r"$F_{\mathrm{e}}$",
           xlim=(eta.min(), eta.max()), ylim=(0, 1))
    ax.set_title("(a) Same input, different memory", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    bins = np.linspace(0, max(0.62, diam.max()), 24)
    ax.hist(diam, bins=bins, color=BLUE, alpha=0.82, edgecolor="white", lw=0.35)
    ax.axvline(diam.mean(), color=ORANGE, lw=1.5,
               label=rf"mean $\Delta={diam.mean():.3f}$")
    ax.text(0.97, 0.92,
            rf"mean $\Delta/2={diam.mean()/2:.3f}$" + "\n"
            + rf"Bayes MAE $={bayes_by_input.mean():.3f}$",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", lw=0.6))
    ax.set(xlabel=r"ambiguity diameter $\Delta_\phi(x)$", ylabel="input groups")
    ax.set_title("(b) Representation ambiguity", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[2]
    ax.errorbar(dfe.per_query_shots, dfe.mae_mean, yerr=dfe.mae_std,
                color=ORANGE, marker="o", ms=3.5, lw=1.45, capsize=2,
                label="stratified DFE")
    ax.errorbar(hybrid.per_query_shots, hybrid.mae_mean, yerr=hybrid.mae_std,
                color=GREEN, marker="s", ms=3.5, lw=1.55, capsize=2,
                label="measurement-conditioned hybrid")
    ax.axhline(prior.mae_mean, color=PURPLE, ls="-.", lw=1.25,
               label="marginal neural prior")
    ax.axhline(bayes_by_input.mean(), color=GREY, ls="--", lw=1.15,
               label="marginal-only Bayes floor")
    ax.annotate("32-shot hybrid\nmatches 64-shot DFE",
                xy=(32, float(hybrid.loc[hybrid.per_query_shots == 32, "mae_mean"].iloc[0])),
                xytext=(7, 0.035), textcoords="offset points", fontsize=6.8,
                arrowprops=dict(arrowstyle="->", lw=0.6, color="0.3"))
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128], labels=[4, 8, 16, 32, 64, 128])
    ax.set(xlabel="projective shots per query", ylabel="MAE",
           ylim=(0.035, 0.295))
    ax.set_title("(c) Measurement-conditioned fusion", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper right", handlelength=2.1)

    for ax in axes:
        ax.grid(alpha=0.16, lw=0.5)
    fig.tight_layout(w_pad=1.15)
    fig.savefig(FIGURES / "identifiability_hybrid.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "identifiability_hybrid.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def amortisation():
    q = np.arange(0, 701)
    fig, ax = plt.subplots(figsize=(4.5, 2.75))
    ax.plot(q, 64 * q, color=ORANGE, lw=1.6, label="DFE, 64 shots/query (MAE 0.0641)")
    ax.plot(q, 6144 + 32 * q, color=GREEN, lw=1.7,
            label="hybrid, 32 shots/query (MAE 0.0657)")
    ax.plot(q, 96 * q, color="#B14A59", lw=1.25, ls="--",
            label="DFE, 96 shots/query (MAE 0.0525)")
    ax.plot(q, 8192 + 64 * q, color=BLUE, lw=1.35, ls="--",
            label="hybrid, 64 shots/query (MAE 0.0539)")
    ax.axvline(192, color="0.4", lw=0.8, ls=":")
    ax.axvline(256, color="0.4", lw=0.8, ls=":")
    ax.text(192, 4500, "192", ha="center", va="bottom", fontsize=7)
    ax.text(256, 4500, "256", ha="center", va="bottom", fontsize=7)
    ax.set(xlabel="deployment queries", ylabel="cumulative projective shots",
           xlim=(0, 700), ylim=(0, 68000))
    ax.grid(alpha=0.18, lw=0.5)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "break_even_queries.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "break_even_queries.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    identifiability_and_hybrid()
    amortisation()
