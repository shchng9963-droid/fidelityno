"""Build the full real-hardware comparison table once the device-regime
checkpoints exist.

Compares:
  - v1 NN ckpts (broad-regime training) on real-hardware data
  - device-regime NN ckpts on real-hardware data
  - product_bound, analytic_best
  - MC at K in {10, 100, 1000}

Writes results_prxq/real_hardware/v2_unified_headline.csv and a
Markdown summary.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


def main() -> None:
    out_root = Path("results_prxq/real_hardware")

    # v1 NN
    v1 = pd.read_csv(out_root / "nn_models" / "headline.csv")
    v1["track"] = "broad_v1"
    v1_pool = v1.groupby(["track", "model", "backend"]).agg(
        mae_F_e=("mae_F_e", "mean"),
        mae_F_e_std=("mae_F_e", "std"),
        ece=("ece", "mean"),
        latency_ms=("latency_ms", "mean"),
    ).reset_index()

    # Device-regime NN (if exists)
    dev_path = out_root / "nn_models_device" / "headline.csv"
    if dev_path.exists():
        dv = pd.read_csv(dev_path)
        dv["track"] = "device_regime"
        dv_pool = dv.groupby(["track", "model", "backend"]).agg(
            mae_F_e=("mae_F_e", "mean"),
            mae_F_e_std=("mae_F_e", "std"),
            ece=("ece", "mean"),
            latency_ms=("latency_ms", "mean"),
        ).reset_index()
    else:
        print(f"[warn] {dev_path} not found yet; running with broad-only.")
        dv_pool = pd.DataFrame(columns=v1_pool.columns)

    nn = pd.concat([v1_pool, dv_pool], ignore_index=True)

    # Analytic + MC
    ab = pd.read_csv(out_root / "analytic_baselines.csv")
    ab_pool = ab[ab.length == -1].copy()

    # Build matrix: row=backend, columns=multiple
    backends = sorted(set(ab_pool.backend) | set(nn.backend))
    rows = []
    for be in backends:
        row = {"backend": be}
        for m in ["product_bound", "analytic_best", "mc_K10", "mc_K100", "mc_K1000"]:
            sub = ab_pool[(ab_pool.backend == be) & (ab_pool.model == m)]
            if len(sub):
                row[m] = float(sub.iloc[0]["mae_F_e"])
        for tr, label_pref in [("broad_v1", "v1"), ("device_regime", "dev")]:
            for m in ["mlp", "deepsets", "bidir", "fidelityno", "fidelityno_large"]:
                sub = nn[(nn.track == tr) & (nn.model == m) & (nn.backend == be)]
                if len(sub):
                    row[f"{label_pref}_{m}"] = float(sub.iloc[0]["mae_F_e"])
                    row[f"{label_pref}_{m}_std"] = float(sub.iloc[0]["mae_F_e_std"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "v2_unified_headline.csv", index=False)

    # Pooled
    pooled = {}
    for col in df.columns:
        if col == "backend" or col.endswith("_std"):
            continue
        try:
            pooled[col] = float(df[col].mean())
        except Exception:
            continue
    pdf = pd.DataFrame([{"model": k, "pooled_MAE_F_e": v} for k, v in pooled.items()])
    pdf = pdf.sort_values("pooled_MAE_F_e").reset_index(drop=True)
    pdf.to_csv(out_root / "v2_pooled_means.csv", index=False)
    print("=== Pooled MAE F_e across 6 backends (lower is better) ===")
    print(pdf.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # Markdown
    lines = ["# Real-hardware MAE on F_e (PRXQ track, post-device-regime retrain)", ""]
    lines.append("| Estimator | MAE_F_e (pooled across 6 IBM backends) |")
    lines.append("|---|---|")
    for _, r in pdf.iterrows():
        lines.append(f"| `{r['model']}` | {r['pooled_MAE_F_e']:.5f} |")
    (out_root / "v2_pooled_means.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
