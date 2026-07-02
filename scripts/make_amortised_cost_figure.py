from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib-sweep", default="results_prxq/collision/calibration_sweep_aggregate.csv")
    ap.add_argument("--dfe", default="results_prxq/collision/dfe_family_ood.csv")
    ap.add_argument("--label-shots-per-seq", type=float, default=6000.0,
                    help="Quantum shots needed to label one OOD calibration sequence. Default follows DFE S=30, M=200.")
    ap.add_argument("--target-mae", type=float, default=0.095,
                    help="Reference target MAE for choosing the matched-accuracy DFE budget.")
    ap.add_argument("--out-dir", default="results_prxq/collision/amortised_cost")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cal = pd.read_csv(args.calib_sweep).sort_values("n_calib")
    dfe = pd.read_csv(args.dfe).sort_values("mae_F_e")

    # pick the nearest DFE operating point to the target MAE
    dfe = dfe.assign(delta=(dfe["mae_F_e"] - args.target_mae).abs())
    dfe_star = dfe.sort_values(["delta", "quantum_shots_per_seq"]).iloc[0]

    # best calibration point that meets / nearly meets target MAE
    cal = cal.assign(delta=(cal["mae_corr_mean"] - args.target_mae).abs())
    cal_star = cal.sort_values(["delta", "n_calib"]).iloc[0]

    n_query = np.unique(np.round(np.logspace(0, 5, 400)).astype(int))
    build_cost = float(cal_star["n_calib"]) * float(args.label_shots_per_seq)
    cost_fidelityno = np.full_like(n_query, build_cost, dtype=float)
    cost_dfe = n_query.astype(float) * float(dfe_star["quantum_shots_per_seq"])

    # break-even where DFE cost overtakes build cost
    break_even_q = int(np.ceil(build_cost / float(dfe_star["quantum_shots_per_seq"])))

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(n_query, cost_fidelityno, lw=2.2, color="#1f77b4",
            label=f"FidelityNO + affine cal (build={int(cal_star['n_calib'])} labels)")
    ax.plot(n_query, cost_dfe, lw=2.0, color="#ff7f0e",
            label=f"DFE matched-accuracy (S={int(dfe_star['S'])}, {int(dfe_star['quantum_shots_per_seq'])} shots/query)")
    ax.axvline(break_even_q, color="#444444", ls="--", lw=1.2)
    ax.scatter([break_even_q], [build_cost], color="#444444", s=24, zorder=3)
    ax.text(break_even_q * 1.05, build_cost * 1.08,
            f"break-even ≈ {break_even_q:,} queries",
            fontsize=9, color="#444444")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Deployment queries")
    ax.set_ylabel("Cumulative quantum shots")
    ax.set_title("Amortised deployment cost in collision family-OOD")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    fig.savefig(out_dir / "break_even_queries.pdf")
    fig.savefig(out_dir / "break_even_queries.png", dpi=180)

    summary = pd.DataFrame([
        {
            "target_mae": float(args.target_mae),
            "fidelityno_n_calib": int(cal_star["n_calib"]),
            "fidelityno_mae": float(cal_star["mae_corr_mean"]),
            "build_cost_shots": float(build_cost),
            "label_shots_per_seq": float(args.label_shots_per_seq),
            "dfe_S": int(dfe_star["S"]),
            "dfe_quantum_shots_per_seq": int(dfe_star["quantum_shots_per_seq"]),
            "dfe_mae": float(dfe_star["mae_F_e"]),
            "break_even_queries": int(break_even_q),
        }
    ])
    summary.to_csv(out_dir / "break_even_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"[saved] {out_dir / 'break_even_queries.pdf'}")


if __name__ == "__main__":
    main()
