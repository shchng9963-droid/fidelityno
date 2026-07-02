"""Aggregate the family-OOD sweep CSVs into a single summary.

Reads results/family_ood_sweep/<model>_holdout_<F>_seed<S>.csv and
emits:
  results/family_ood_sweep/aggregate.csv       (one row per (model,holdout,split,length,seed))
  results/family_ood_sweep/heatmap.csv         ((model x holdout) family-OOD MAE mean+/-std)
  results/figs/family_ood_heatmap.{pdf,png}    Matplotlib heatmap

Also includes the original main-run pauli holdout (results/<model>_seed*.csv) so
the heatmap covers all 5 single-qubit families.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Original main run uses pauli as the held-out family. The newest seed-x-model
# CSVs live under results/main_run/ (the top-level results/<model>_seed*.csv are
# stale precursors); we read main_run/ to match Table~tab:famood in the paper.
MAIN_RUN_HOLDOUT = "pauli"
MAIN_RUN_DIR = Path("results/main_run")
SWEEP_DIR = Path("results/family_ood_sweep")
FIGS_DIR = Path("results/figs")

# Models we want to display, in row order.
MODELS = ["fidelityno", "gnn", "generic_gnn", "mlp", "deepsets"]
PRETTY = {
    "fidelityno": "FidelityNO-T",
    "gnn": "FidelityNO-G",
    "generic_gnn": "PathGNN (no Choi enc.)",
    "mlp": "FlatMLP",
    "deepsets": "DeepSets",
}
HOLDOUTS = ["amplitude_damping", "phase_damping", "depolarizing", "pauli", "lindblad"]
PRETTY_FAM = {
    "amplitude_damping": "Amp.Damp.",
    "phase_damping":     "Phase Damp.",
    "depolarizing":      "Depol.",
    "pauli":             "Pauli",
    "lindblad":          "Lindblad",
}


SWEEP_RX = re.compile(r"^(?P<model>[a-z_]+?)_holdout_(?P<holdout>[a-z_]+)_seed(?P<seed>\d+)\.csv$")
MAIN_RX  = re.compile(r"^(?P<model>[a-z_]+?)_seed(?P<seed>\d+)\.csv$")


def load_sweep_rows() -> pd.DataFrame:
    rows = []
    for csv in sorted(SWEEP_DIR.glob("*.csv")):
        m = SWEEP_RX.match(csv.name)
        if not m:
            continue
        df = pd.read_csv(csv)
        df["holdout"] = m.group("holdout")
        df["model"]   = m.group("model")        # eval.py also writes 'model'; trust filename
        df["seed"]    = int(m.group("seed"))
        df["source"]  = "sweep"
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_main_run_pauli_rows() -> pd.DataFrame:
    """Pull the original (pauli-holdout) results for the 5 models we care about."""
    rows = []
    for model in MODELS:
        for csv in sorted(MAIN_RUN_DIR.glob(f"{model}_seed*.csv")):
            m = MAIN_RX.match(csv.name)
            if not m:
                continue
            # Exclude the *_calibrated variants.
            if "calibrated" in csv.name:
                continue
            df = pd.read_csv(csv)
            df["holdout"] = MAIN_RUN_HOLDOUT
            df["model"]   = model
            df["seed"]    = int(m.group("seed"))
            df["source"]  = "main_run"
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the family_ood split.

    To match the convention used in Table~tab:famood (mean ± std reported
    on the family-OOD test set across both seeds and the three test
    lengths), we compute mean and std *jointly* over all (seed, length)
    rows for each (model, holdout) cell, not seed-only. The seed-only
    breakdown is preserved separately in per_seed.
    """
    sub = df[df["split"] == "family_ood"].copy()
    # Joint (seed × length) mean ± std — paper convention.
    summary = sub.groupby(["model", "holdout"], as_index=False).agg(
        mae_mean=("mae", "mean"),
        mae_std =("mae", "std"),
        pinball_mean=("pinball", "mean"),
        pinball_std =("pinball", "std"),
        crps_mean=("crps", "mean"),
        crps_std =("crps", "std"),
        ece_mean=("ece", "mean"),
        ece_std =("ece", "std"),
        n_rows  =("mae", "count"),
        n_seeds =("seed", "nunique"),
    )
    # Per-seed (length-averaged) for diagnostics.
    per_seed = sub.groupby(["model", "holdout", "seed"], as_index=False).agg(
        mae=("mae", "mean"),
        pinball=("pinball", "mean"),
        crps=("crps", "mean"),
        ece=("ece", "mean"),
    )
    return summary, per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-main-run", action="store_true",
                    help="Also fold in main-run pauli-holdout numbers as the 5th column.")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    sweep = load_sweep_rows()
    print(f"Loaded {len(sweep)} sweep rows from {SWEEP_DIR}")
    if args.include_main_run:
        main_run = load_main_run_pauli_rows()
        print(f"Loaded {len(main_run)} main-run rows for pauli holdout")
        all_rows = pd.concat([sweep, main_run], ignore_index=True) if len(main_run) else sweep
    else:
        all_rows = sweep

    SWEEP_DIR.mkdir(exist_ok=True, parents=True)
    all_rows.to_csv(SWEEP_DIR / "raw_combined.csv", index=False)
    print(f"Wrote {SWEEP_DIR/'raw_combined.csv'}  rows={len(all_rows)}")

    summary, per_seed = aggregate(all_rows)
    summary.to_csv(SWEEP_DIR / "aggregate.csv", index=False)
    per_seed.to_csv(SWEEP_DIR / "per_seed.csv", index=False)

    # Pretty-print the (model x holdout) family-OOD MAE table.
    pivot_mean = summary.pivot(index="model", columns="holdout", values="mae_mean")
    pivot_std  = summary.pivot(index="model", columns="holdout", values="mae_std")

    # Restrict + reorder.
    cols_present = [h for h in HOLDOUTS if h in pivot_mean.columns]
    rows_present = [m for m in MODELS   if m in pivot_mean.index]
    pivot_mean = pivot_mean.loc[rows_present, cols_present]
    pivot_std  = pivot_std .loc[rows_present, cols_present]

    print("\nFamily-OOD MAE (mean across seeds, averaged over lengths in family_ood):")
    fmt = pd.DataFrame(index=pivot_mean.index, columns=pivot_mean.columns, dtype=object)
    for r in fmt.index:
        for c in fmt.columns:
            mu = pivot_mean.loc[r, c]
            sd = pivot_std.loc[r, c] if not pd.isna(pivot_std.loc[r, c]) else 0.0
            fmt.loc[r, c] = f"{mu:.4f}±{sd:.4f}" if not pd.isna(mu) else "—"
    print(fmt.to_string())

    # Heatmap CSV.
    pivot_mean.to_csv(SWEEP_DIR / "heatmap_mean.csv")
    pivot_std .to_csv(SWEEP_DIR / "heatmap_std.csv")

    if args.no_fig:
        return

    # ----- Plot -----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS_DIR.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(1.0 + 1.4 * len(cols_present), 0.7 + 0.55 * len(rows_present)))
    Z = pivot_mean.values.astype(float)
    cmap = plt.cm.get_cmap("magma_r")
    vmin = np.nanmin(Z)
    vmax = np.nanmax(Z)
    im = ax.imshow(Z, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(cols_present)), [PRETTY_FAM.get(c, c) for c in cols_present], rotation=20)
    ax.set_yticks(range(len(rows_present)), [PRETTY.get(m, m) for m in rows_present])
    ax.set_xlabel("Held-out family (Family-OOD test)")
    ax.set_ylabel("Model")
    for i, r in enumerate(rows_present):
        for j, c in enumerate(cols_present):
            mu = pivot_mean.loc[r, c]
            sd = pivot_std .loc[r, c] if not pd.isna(pivot_std.loc[r, c]) else 0.0
            if pd.isna(mu):
                txt = "—"
            else:
                txt = f"{mu:.3f}\n±{sd:.3f}"
            color = "white" if (mu - vmin) / max(vmax - vmin, 1e-9) > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    ax.set_title("Family-OOD MAE: rows = model, cols = held-out noise family")
    fig.colorbar(im, ax=ax, label="MAE")
    fig.tight_layout()
    out_pdf = FIGS_DIR / "family_ood_heatmap.pdf"
    out_png = FIGS_DIR / "family_ood_heatmap.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    print(f"\nWrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
