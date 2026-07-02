"""Deeper-than-pooled diagnostic: how does each model's MAE scale with
sequence length, and where does FidelityNO start to beat the analytic
product bound?

Reads results_prxq/real_hardware/{nn_models/by_length.csv, analytic_baselines.csv}
and writes a per-length comparison + a CSV.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np


def main() -> None:
    out_root = Path("results_prxq/real_hardware")

    # NN per-length
    nn = pd.read_csv(out_root / "nn_models" / "by_length.csv")
    nn_by_len = nn.groupby(["model", "backend", "length"]).agg(
        mae_F_e=("mae_F_e", "mean"),
        mae_F_e_std=("mae_F_e", "std"),
    ).reset_index()

    # Analytic + MC per length
    ab = pd.read_csv(out_root / "analytic_baselines.csv")
    ab_by_len = ab[ab.length != -1].copy()

    # Combined per length
    ab_lens = ab_by_len.pivot_table(index=["backend", "length"], columns="model",
                                     values="mae_F_e").reset_index()
    nn_lens = nn_by_len.pivot_table(index=["backend", "length"], columns="model",
                                    values="mae_F_e").reset_index()
    both = ab_lens.merge(nn_lens, on=["backend", "length"], how="outer")
    both = both.sort_values(["backend", "length"])
    both.to_csv(out_root / "by_length.csv", index=False)

    # Average across backends to a length-curve table
    cols_keep = ["product_bound", "analytic_best",
                 "mc_K100", "mc_K1000",
                 "deepsets", "fidelityno", "fidelityno_large"]
    cols_keep = [c for c in cols_keep if c in both.columns]
    avg = both.groupby("length")[cols_keep].mean().reset_index()
    avg.to_csv(out_root / "length_curve.csv", index=False)
    print("=== Length curves (mean MAE_F_e across 6 backends) ===")
    print(avg.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # Where does FidelityNO beat product?
    print("\n=== Where FidelityNO_large vs product_bound? ===")
    # Per backend per length
    for be in sorted(both.backend.unique()):
        sub = both[both.backend == be].sort_values("length")
        if "fidelityno_large" not in sub.columns or "product_bound" not in sub.columns:
            continue
        rows = []
        for _, r in sub.iterrows():
            rows.append((int(r.length), float(r["product_bound"]), float(r["fidelityno_large"])))
        print(f"  {be}:  ", " ".join(f"L={L} prod={p:.4f} FNO_L={f:.4f}" for L, p, f in rows))


if __name__ == "__main__":
    main()
