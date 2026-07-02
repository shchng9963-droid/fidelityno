"""Build the headline figure + table comparing FidelityNO (zero quantum
shots) to DFE at multiple sample budgets, on real-hardware data.

Inputs:
  - results_prxq/real_hardware/nn_models_device/headline.csv
  - results_prxq/dfe/dfe_real_hardware.csv

Outputs:
  - results_prxq/dfe/sample_complexity.csv
  - results_prxq/dfe/sample_complexity.pdf
  - results_prxq/dfe/sample_complexity.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    nn_path = Path("results_prxq/real_hardware/nn_models_device/headline.csv")
    dfe_path = Path("results_prxq/dfe/dfe_real_hardware.csv")
    out_root = Path("results_prxq/dfe")
    out_root.mkdir(parents=True, exist_ok=True)

    if not dfe_path.exists():
        print(f"[error] missing {dfe_path}; run scripts/eval_dfe.py first.")
        return

    dfe = pd.read_csv(dfe_path)
    dfe_pool = dfe.groupby("S").agg(
        mae_F_e_mean=("mae_F_e", "mean"),
        mae_F_e_std=("mae_F_e", "std"),
        shots=("quantum_shots_per_seq", "mean"),
    ).reset_index()
    dfe_pool["model"] = "DFE"

    rows = [{
        "model": f"DFE_S={int(r.S)}",
        "shots_per_seq": float(r.shots),
        "mae_F_e": float(r.mae_F_e_mean),
        "mae_F_e_std": float(r.mae_F_e_std) if not np.isnan(r.mae_F_e_std) else 0.0,
    } for r in dfe_pool.itertuples()]

    if nn_path.exists():
        nn = pd.read_csv(nn_path)
        for model in sorted(nn.model.unique()):
            sub = nn[nn.model == model]
            rows.append({
                "model": f"FidelityNO/{model} (device-regime)",
                "shots_per_seq": 0.0,
                "mae_F_e": float(sub.mae_F_e.mean()),
                "mae_F_e_std": float(sub.mae_F_e.std()) if len(sub) > 1 else 0.0,
            })
    else:
        print(f"[warn] {nn_path} missing; plotting DFE alone.")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / "sample_complexity.csv", index=False)

    # Plot: x = shots/seq, y = MAE; FidelityNO at x=0 marker; DFE as a curve.
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    dfe_curve = summary[summary.model.str.startswith("DFE_")].copy()
    dfe_curve = dfe_curve.sort_values("shots_per_seq")
    ax.errorbar(dfe_curve["shots_per_seq"], dfe_curve["mae_F_e"],
                yerr=dfe_curve["mae_F_e_std"], marker="o", capsize=3,
                color="#2c7fb8", label="DFE (S Paulis x 200 shots)")
    nn_rows = summary[~summary.model.str.startswith("DFE_")]
    palette = {"fidelityno": "#d95f02",
               "fidelityno_large": "#1b9e77",
               "mlp": "#7570b3",
               "deepsets": "#a6761d",
               "bidir": "#e6ab02"}
    for r in nn_rows.itertuples():
        m_short = r.model.split("/")[-1].split(" ")[0]
        c = palette.get(m_short, "#666666")
        ax.errorbar([1.0], [r.mae_F_e], yerr=[r.mae_F_e_std], marker="*",
                    markersize=14, capsize=3, color=c,
                    label=f"{m_short} (zero shots)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Quantum shots per sequence")
    ax.set_ylabel("MAE on entanglement fidelity F_e")
    ax.set_title("Sample-complexity comparison (5 IBM backends, n_eval=1024)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_root / "sample_complexity.pdf")
    fig.savefig(out_root / "sample_complexity.png", dpi=150)
    plt.close(fig)
    print(f"[saved] {out_root/'sample_complexity.pdf'}")

    # MD table
    lines = ["# Sample-complexity comparison: DFE vs FidelityNO (device-regime)", ""]
    lines.append("| Estimator | shots/seq | MAE F_e | std |")
    lines.append("|---|---|---|---|")
    for r in summary.sort_values("shots_per_seq").itertuples():
        lines.append(f"| {r.model} | {r.shots_per_seq:.0f} | {r.mae_F_e:.5f} | {r.mae_F_e_std:.5f} |")
    (out_root / "sample_complexity.md").write_text("\n".join(lines))
    print(f"[saved] {out_root/'sample_complexity.md'}")


if __name__ == "__main__":
    main()
