"""Build the unified real-hardware headline table.

Combines:
  - Analytic baselines (product_bound, fvg_bound, analytic_best)
  - Monte Carlo at K in {10, 100, 1000}
  - All trained NN checkpoints (averaged over 5 seeds)

Writes results_prxq/real_hardware/unified_headline.csv plus prints.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


def main() -> None:
    out_root = Path("results_prxq/real_hardware")
    ab = pd.read_csv(out_root / "analytic_baselines.csv")
    ab_pool = ab[ab.length == -1].copy()

    nn = pd.read_csv(out_root / "nn_models" / "headline.csv")
    nn_pool = nn.groupby(["model", "backend"]).agg(
        mae_F_e=("mae_F_e", "mean"),
        mae_F_e_std=("mae_F_e", "std"),
        mae_F_avg=("mae_F_avg", "mean"),
        ece=("ece", "mean"),
        latency_ms=("latency_ms", "mean"),
    ).reset_index()
    nn_pool.to_csv(out_root / "nn_models" / "by_model_backend.csv", index=False)

    rows = []
    backends = sorted(set(ab_pool.backend) | set(nn_pool.backend))
    for be in backends:
        row = {"backend": be}
        for model in ["product_bound", "fvg_bound", "analytic_best",
                      "mc_K10", "mc_K100", "mc_K1000"]:
            sub = ab_pool[(ab_pool.backend == be) & (ab_pool.model == model)]
            if len(sub):
                row[model] = float(sub.iloc[0]["mae_F_e"])
        for model in ["mlp", "gnn", "generic_gnn", "deepsets", "bidir",
                      "fidelityno", "fidelityno_large"]:
            sub = nn_pool[(nn_pool.backend == be) & (nn_pool.model == model)]
            if len(sub):
                row[model] = float(sub.iloc[0]["mae_F_e"])
                row[f"{model}_std"] = float(sub.iloc[0]["mae_F_e_std"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "unified_headline.csv", index=False)

    # Print summary - F_e MAE, headline columns only
    cols_mae = [
        "backend",
        "product_bound", "analytic_best",
        "mc_K10", "mc_K100", "mc_K1000",
        "mlp", "gnn", "deepsets", "bidir",
        "fidelityno", "fidelityno_large",
    ]
    print("=== Real-hardware MAE on F_e (lower is better) ===")
    print(df[cols_mae].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Latency table (ms per sequence) - indicative
    lat_rows = []
    for be in backends:
        row = {"backend": be}
        for model in ["product_bound", "mc_K10", "mc_K100", "mc_K1000"]:
            sub = ab_pool[(ab_pool.backend == be) & (ab_pool.model == model)]
            if len(sub):
                row[model] = float(sub.iloc[0]["latency_ms"])
        for model in ["fidelityno", "fidelityno_large"]:
            sub = nn_pool[(nn_pool.backend == be) & (nn_pool.model == model)]
            if len(sub):
                row[model] = float(sub.iloc[0]["latency_ms"])
        lat_rows.append(row)
    lat = pd.DataFrame(lat_rows)
    lat.to_csv(out_root / "unified_latency.csv", index=False)
    print("\n=== Latency (ms per sequence) ===")
    print(lat.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Pooled across backends
    print("\n=== Pooled across 6 backends (overall MAE_F_e mean) ===")
    pooled = {}
    for col in cols_mae:
        if col == "backend":
            continue
        pooled[col] = float(df[col].mean())
    pdf = pd.DataFrame([pooled]).T.rename(columns={0: "MAE_F_e_pooled"})
    pdf.to_csv(out_root / "pooled_means.csv")
    print(pdf.sort_values("MAE_F_e_pooled").to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
