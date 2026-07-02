"""Build the central PRXQ paper figure (3-panel sample complexity).

Panels (left -> right):
  (a) Device regime  (IBM Fake*V2 hardware-realistic noise; d=2)
  (b) Non-Markovian collision channels   (d=2, eta-mediated bath retention)
  (c) Two-qubit order-sensitive channels (d=4)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results_prxq" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


CAT_COLOR = {
    "analytic": "tab:blue",
    "MC": "tab:gray",
    "DFE": "tab:green",
    "NN": "tab:red",
    "NN+cal": "tab:purple",
}
CAT_MARKER = {"analytic": "s", "MC": "D", "DFE": "^", "NN": "o", "NN+cal": "*"}


# ---------- per-panel data extractors ----------

def _device_points():
    pts = []
    f = ROOT / "results_prxq" / "real_hardware" / "v2_pooled_means.csv"
    if f.exists():
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            label = row.get("model") or row.get("estimator") or "?"
            mae = float(
                row.get("pooled_MAE_F_e")
                or row.get("mae_F_e_pooled")
                or row.get("mae_F_e", np.nan)
            )
            shots = float(row.get("shots_per_seq", 0.0))
            # derive shots from MC model names: mc_K10 -> 10, mc_K1000 -> 1000
            if shots == 0.0 and "K" in str(label):
                try:
                    shots = float(str(label).split("K")[-1])
                except Exception:
                    pass
            cat = _classify(label)
            pts.append((cat, label, shots, mae, 0.0))
    # always also append DFE points
    f = ROOT / "results_prxq" / "dfe" / "dfe_real_hardware.csv"
    if f.exists():
        df = pd.read_csv(f).dropna(subset=["mae_F_e"])
        for S, sub in df.groupby("S"):
            shots = float(sub.quantum_shots_per_seq.mean()) if "quantum_shots_per_seq" in sub.columns else float(S) * 200
            pts.append(("DFE", f"S={int(S)}", shots, float(sub.mae_F_e.mean()), 0.0))
    return pts


def _collision_points(split="id_test"):
    pts = []
    base = ROOT / "results_prxq" / "collision"
    f = base / "analytic.csv"
    if f.exists():
        df = pd.read_csv(f)
        df = df[df.split == split]
        for m, sub in df.groupby("model"):
            pts.append(("analytic", m, 0.0, float(sub.mae.mean()), 0.0))
    f = base / "mc.csv"
    if f.exists():
        df = pd.read_csv(f)
        df = df[df.split == split]
        for m, sub in df.groupby("model"):
            try:
                K = int(m.split("_")[-1])
            except Exception:
                K = 1
            pts.append(("MC", m, float(K), float(sub.mae.mean()), 0.0))
    f = base / "dfe.csv"
    if f.exists():
        df = pd.read_csv(f).dropna(subset=["mae_F_e"])
        df = df[df.split == split]
        for S, sub in df.groupby("S"):
            pts.append(("DFE", f"S={int(S)}", float(S) * 200, float(sub.mae_F_e.mean()), 0.0))
    for f in sorted(base.glob("*_seed*.csv")):
        if f.name.startswith(("recalibrated", "summary", "pooled")):
            continue
        df = pd.read_csv(f)
        if "model" not in df.columns:
            continue
        df = df[df.split == split]
        m = str(df["model"].iloc[0]) if len(df) else f.stem
        # only one point per architecture/seed; we will pool below
        pts.append(("NN", f"{m}_{f.stem.split('_')[-1]}", 0.0, float(df.mae.mean()), 0.0))
    for f in sorted(base.glob("recalibrated_*.csv")):
        m = f.stem.replace("recalibrated_", "")
        df = pd.read_csv(f)
        df = df[df.split == split]
        if not len(df):
            continue
        pts.append(("NN+cal", m, 0.0, float(df.mae_F_e_corr.mean()), float(df.mae_F_e_corr.std())))
    return pts


def _two_qubit_d4_points(split="id_test"):
    pts = []
    base = ROOT / "results_prxq" / "two_qubit_d4"
    f = base / "analytic.csv"
    if f.exists():
        df = pd.read_csv(f)
        df = df[df.split == split]
        for m, sub in df.groupby("model"):
            pts.append(("analytic", m, 0.0, float(sub.mae.mean()), 0.0))
    f = base / "mc.csv"
    if f.exists():
        df = pd.read_csv(f)
        df = df[df.split == split]
        for m, sub in df.groupby("model"):
            try:
                K = int(m.split("_")[-1])
            except Exception:
                K = 1
            pts.append(("MC", m, float(K), float(sub.mae.mean()), 0.0))
    for f in sorted(base.glob("recalibrated_*.csv")):
        m = f.stem.replace("recalibrated_", "")
        df = pd.read_csv(f)
        df = df[df.split == split]
        if not len(df):
            continue
        pts.append(("NN+cal", m, 0.0, float(df.mae_F_e_corr.mean()), float(df.mae_F_e_corr.std())))
    return pts


def _classify(label):
    s = label.lower()
    if any(k in s for k in ("product_bound", "fvg", "diamond", "analytic")):
        return "analytic"
    if "mc" in s:
        return "MC"
    if "dfe" in s:
        return "DFE"
    if "+cal" in s:
        return "NN+cal"
    return "NN"


# ---------- plotting ----------

def plot_panel(ax, points, title):
    plotted_cats = set()
    for cat, label, shots, mae, std in points:
        if mae <= 0 or not np.isfinite(mae):
            continue
        c = CAT_COLOR.get(cat, "tab:gray")
        m = CAT_MARKER.get(cat, "x")
        x = max(shots, 1.0)
        kw = dict(marker=m, color=c, ms=8 if cat == "NN+cal" else 6,
                  lw=0, elinewidth=1, capsize=2)
        if cat not in plotted_cats:
            kw["label"] = cat
            plotted_cats.add(cat)
        ax.errorbar(x, mae, yerr=std, **kw)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Quantum shots / sequence (1 = zero-shot)")
    ax.set_ylabel(r"Pooled MAE $|F_e^{\rm pred} - F_e^{\rm true}|$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="best")


def main():
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(16, 5))
    plot_panel(axL, _device_points(), "(a) Device regime (d=2)")
    plot_panel(axM, _collision_points("id_test"), "(b) Collision non-Markovian (d=2, id-test)")
    plot_panel(axR, _two_qubit_d4_points("id_test"), "(c) Two-qubit order-sensitive (d=4, id-test)")
    plt.tight_layout()
    fig.savefig(OUT / "central_sample_complexity.pdf")
    fig.savefig(OUT / "central_sample_complexity.png", dpi=140)
    print(f"[saved] {OUT/'central_sample_complexity.pdf'}")


if __name__ == "__main__":
    main()
