"""Build the final pooled non-Markovian collision results table."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

base = Path("results_prxq/collision")
splits = ["id_test", "length_ood", "family_ood"]

print("=" * 80)
print("FINAL POOLED RESULTS — Non-Markovian collision dataset")
print("  (5 seeds per model, 8000 sequences per split)")
print("=" * 80)
hdr = f'{"Estimator":<28s} {"shots":>10s} ' + "  ".join(f'{s:>11s}' for s in splits)
print()
print(hdr)
print("-" * len(hdr))

# Analytic
adf = pd.read_csv(base / "analytic.csv")
for m in sorted(adf.model.unique()):
    sub = adf[adf.model == m]
    nums = [sub[sub.split == s].mae.mean() if (sub.split == s).any() else float("nan") for s in splits]
    print(f"{m:<28s} {0:>10d} " + "  ".join(f'{x:>11.4f}' for x in nums))

# MC
mc = pd.read_csv(base / "mc.csv")
for m in sorted(mc.model.unique()):
    sub = mc[mc.model == m]
    K = m.split("_K")[-1] if "_K" in m else "inf"
    nums = [sub[sub.split == s].mae.mean() if (sub.split == s).any() else float("nan") for s in splits]
    print(f"{m:<28s} {K:>10s} " + "  ".join(f'{x:>11.4f}' for x in nums))

# DFE
dfe = pd.read_csv(base / "dfe.csv").dropna(subset=["mae_F_e"])
for S in sorted(dfe.S.unique()):
    sub = dfe[dfe.S == S]
    name = f"DFE_S{int(S)}"
    nums = [sub[sub.split == s].mae_F_e.mean() if (sub.split == s).any() else float("nan") for s in splits]
    print(f"{name:<28s} {int(S):>10d} " + "  ".join(f'{x:>11.4f}' for x in nums))

print()
print("-- raw NN predictions (no calibration) --")
# Raw NN
nn_csvs = sorted(base.glob("*_seed*.csv"))
nn_csvs = [c for c in nn_csvs if not c.name.startswith(("recalibrated", "summary", "pooled"))]
data = []
for f in nn_csvs:
    df = pd.read_csv(f)
    if "model" in df.columns:
        data.append(df)
if data:
    allnn = pd.concat(data)
    for m in sorted(allnn.model.unique()):
        sub = allnn[allnn.model == m]
        name = f"{m}_raw"
        nums = [sub[sub.split == s].mae.mean() if (sub.split == s).any() else float("nan") for s in splits]
        print(f"{name:<28s} {0:>10d} " + "  ".join(f'{x:>11.4f}' for x in nums))

print()
print("-- post-hoc affine recalibration on 10% of OOD samples --")
for f in sorted(base.glob("recalibrated_*.csv")):
    m = f.stem.replace("recalibrated_", "")
    df = pd.read_csv(f)
    name = f"{m}_cal"
    nums = [df[df.split == s].mae_F_e_corr.mean() if (df.split == s).any() else float("nan") for s in splits]
    stds = [df[df.split == s].mae_F_e_corr.std() if (df.split == s).any() else 0.0 for s in splits]
    print(f"{name:<28s} {0:>10d} " + "  ".join(f'{x:>11.4f}' for x in nums))
