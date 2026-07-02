"""Build the unified PRXQ paper-figure comparing all estimators on
both regimes (device-regime IBM noise vs non-Markovian collision data).

Inputs:
  results_prxq/real_hardware/nn_models_device/headline.csv     (device, NN)
  results_prxq/real_hardware/analytic_baselines.csv            (device, analytic+MC)
  results_prxq/dfe/dfe_real_hardware.csv                       (device, DFE)
  results_prxq/collision/analytic.csv                          (collision, analytic)
  results_prxq/collision/mc.csv                                (collision, MC)
  results_prxq/collision/dfe_<split>.csv                       (collision, DFE)
  results_prxq/collision/<model>_seed<i>.csv                   (collision, NN)

Output:
  results_prxq/figures/sample_complexity_combined.pdf
  results_prxq/figures/sample_complexity_combined.png
  results_prxq/figures/regime_comparison_table.md
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


def load_device() -> dict[str, list[tuple[float, float, float]]]:
    """Return dict {estimator: list of (shots, mae, std)}."""
    out: dict[str, list[tuple[float, float, float]]] = {}

    # NN models (device-regime training only; pooled over backends)
    p_nn = Path("results_prxq/real_hardware/nn_models_device/headline.csv")
    if p_nn.exists():
        nn = pd.read_csv(p_nn)
        for m in sorted(nn.model.unique()):
            sub = nn[nn.model == m]
            mae = float(sub.mae_F_e.mean())
            std = float(sub.mae_F_e.std()) if len(sub) > 1 else 0.0
            out[f"NN/{m}"] = [(0.0, mae, std)]

    # Analytic + MC (pooled across 6 backends, length=-1 = pooled)
    p_an = Path("results_prxq/real_hardware/analytic_baselines.csv")
    if p_an.exists():
        an = pd.read_csv(p_an)
        an_pool = an[an.length == -1].copy()
        for model_name, label, shots in [
            ("product_bound", "analytic/product_bound", 0.0),
            ("analytic_best", "analytic/best", 0.0),
            ("mc_K10", "MC/K=10", 10.0),
            ("mc_K100", "MC/K=100", 100.0),
            ("mc_K1000", "MC/K=1000", 1000.0),
        ]:
            sub = an_pool[an_pool.model == model_name]
            if len(sub) == 0:
                continue
            mae = float(sub.mae_F_e.mean())
            std = float(sub.mae_F_e.std()) if len(sub) > 1 else 0.0
            out[label] = [(shots, mae, std)]

    # DFE
    p_dfe = Path("results_prxq/dfe/dfe_real_hardware.csv")
    if p_dfe.exists():
        df = pd.read_csv(p_dfe)
        for S in sorted(df.S.unique()):
            sub = df[df.S == S]
            shots = float(sub.quantum_shots_per_seq.mean())
            mae = float(sub.mae_F_e.mean())
            std = float(sub.mae_F_e.std()) if len(sub) > 1 else 0.0
            out[f"DFE/S={int(S)}"] = [(shots, mae, std)]

    return out


def load_collision(split: str) -> dict[str, list[tuple[float, float, float]]]:
    out: dict[str, list[tuple[float, float, float]]] = {}

    # Analytic
    p_an = Path("results_prxq/collision/analytic.csv")
    if p_an.exists():
        an = pd.read_csv(p_an)
        an_split = an[an.split == split]
        for model_name, label, shots in [
            ("product_bound", "analytic/product_bound", 0.0),
            ("analytic_best", "analytic/best", 0.0),
        ]:
            sub = an_split[an_split.model == model_name]
            if len(sub):
                out[label] = [(shots, float(sub.mae.mean()), 0.0)]

    # MC
    p_mc = Path("results_prxq/collision/mc.csv")
    if p_mc.exists():
        mc = pd.read_csv(p_mc)
        mc_split = mc[mc.split == split]
        for model_name, K in [("mc_10", 10), ("mc_100", 100), ("mc_1000", 1000)]:
            sub = mc_split[mc_split.model == model_name]
            if len(sub):
                out[f"MC/K={K}"] = [(float(K), float(sub.mae.mean()), 0.0)]

    # DFE (per-split file)
    p_dfe = Path(f"results_prxq/collision/dfe_{split}.csv")
    if p_dfe.exists():
        df = pd.read_csv(p_dfe)
        for S in sorted(df.S.unique()):
            sub = df[df.S == S]
            shots = float(sub.quantum_shots_per_seq.mean())
            mae = float(sub.mae_F_e.mean())
            out[f"DFE/S={int(S)}"] = [(shots, mae, 0.0)]

    # NN models (raw + recalibrated)
    rows = []
    for f in sorted(Path("results_prxq/collision").glob("*_seed*.csv")):
        if f.name.startswith(("recalibrated", "summary", "pooled")):
            continue
        df = pd.read_csv(f)
        if "model" not in df.columns or "split" not in df.columns:
            continue
        sub = df[df.split == split]
        if "mae" in sub.columns:
            mae_col = "mae"
        elif "mae_F_e" in sub.columns:
            mae_col = "mae_F_e"
        else:
            continue
        rows.append({"model": df["model"].iloc[0],
                     "seed": df.get("seed", pd.Series([0])).iloc[0],
                     "mae": float(sub[mae_col].mean())})
    if rows:
        nn_df = pd.DataFrame(rows)
        for m in sorted(nn_df.model.unique()):
            sub = nn_df[nn_df.model == m]
            mae = float(sub.mae.mean())
            std = float(sub.mae.std()) if len(sub) > 1 else 0.0
            out[f"NN/{m}"] = [(0.0, mae, std)]

    # Recalibrated NN
    for csv_path in sorted(Path("results_prxq/collision").glob("recalibrated_*.csv")):
        m = csv_path.stem.replace("recalibrated_", "")
        df = pd.read_csv(csv_path)
        sub = df[df.split == split]
        if len(sub) == 0:
            continue
        mae = float(sub.mae_F_e_corr.mean())
        std = float(sub.mae_F_e_corr.std()) if len(sub) > 1 else 0.0
        out[f"NN+cal/{m}"] = [(0.0, mae, std)]

    return out


def plot_panel(ax, data: dict, title: str, xmax: float = 3e5) -> None:
    # Sort estimators into three groups: NN (zero shots), analytic (zero
    # shots), DFE (S>0), MC (K samples).
    nn_color = {
        "fidelityno": "#d95f02",
        "fidelityno_large": "#1b9e77",
        "mlp": "#7570b3",
        "deepsets": "#a6761d",
        "bidir": "#e6ab02",
    }
    dfe_pts = []
    mc_pts = []
    for label, pts in data.items():
        for shots, mae, std in pts:
            if label.startswith("DFE/"):
                dfe_pts.append((shots, mae, std, label))
            elif label.startswith("MC/"):
                mc_pts.append((shots, mae, std, label))
            elif label.startswith("analytic/"):
                ax.axhline(mae, ls="--", lw=1.0, color="#444444",
                           label=f"{label} (zero shots)" if "best" in label else None)
            elif label.startswith("NN/"):
                m = label.split("/")[-1]
                c = nn_color.get(m, "#888888")
                # plot at shots=1 so it sits to the left on log scale
                ax.errorbar([1.0], [mae], yerr=[std], marker="*", markersize=14,
                            capsize=3, color=c, label=f"{m} (zero shots)")
    if dfe_pts:
        dfe_pts.sort()
        xs = [p[0] for p in dfe_pts]
        ys = [p[1] for p in dfe_pts]
        ax.plot(xs, ys, marker="o", ms=6, color="#2c7fb8", lw=1.8, label="DFE (S Paulis × 200 shots)")
    if mc_pts:
        mc_pts.sort()
        xs = [p[0] for p in mc_pts]
        ys = [p[1] for p in mc_pts]
        ax.plot(xs, ys, marker="^", ms=6, color="#cc4c02", lw=1.5, label="MC Kraus sampling (K)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("quantum shots per sequence")
    ax.set_ylabel("MAE on F_e")
    ax.set_title(title)
    ax.set_xlim(0.7, xmax)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best")


def main() -> None:
    out_root = Path("results_prxq/figures")
    out_root.mkdir(parents=True, exist_ok=True)

    device = load_device()
    coll_id = load_collision("id_test")
    coll_lood = load_collision("length_ood")
    coll_food = load_collision("family_ood")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharey=False)
    plot_panel(axes[0], device, "Device-regime IBM noise\n(5 backends, pooled)")
    plot_panel(axes[1], coll_id, "Non-Markovian collision\nid_test (in-distribution)")
    plot_panel(axes[2], coll_lood, "Non-Markovian collision\nlength_ood (L 24-48)")
    plot_panel(axes[3], coll_food, "Non-Markovian collision\nfamily_ood (eta 0.85-0.99)")
    fig.tight_layout()
    fig.savefig(out_root / "sample_complexity_combined.pdf")
    fig.savefig(out_root / "sample_complexity_combined.png", dpi=160)
    plt.close(fig)
    print(f"[saved] {out_root/'sample_complexity_combined.pdf'}")
    print(f"[saved] {out_root/'sample_complexity_combined.png'}")

    # Markdown comparison table
    lines = ["# PRXQ unified comparison", ""]
    for tag, data in [
        ("Device-regime IBM noise", device),
        ("Non-Markovian collision: id_test", coll_id),
        ("Non-Markovian collision: length_ood", coll_lood),
        ("Non-Markovian collision: family_ood", coll_food),
    ]:
        lines.append(f"## {tag}\n")
        lines.append("| Estimator | shots/seq | MAE F_e | std |")
        lines.append("|---|---|---|---|")
        for k in sorted(data.keys()):
            for shots, mae, std in data[k]:
                lines.append(f"| {k} | {shots:.0f} | {mae:.5f} | {std:.5f} |")
        lines.append("")
    (out_root / "regime_comparison_table.md").write_text("\n".join(lines))
    print(f"[saved] {out_root/'regime_comparison_table.md'}")


if __name__ == "__main__":
    main()
