"""Build the unified PRXQ P1.1 (non-Markovian collision) summary table.

Aggregates analytic, MC, DFE, and NN model results from results_prxq/collision/
into a single pooled-MAE-by-split table that matches the device-regime
table format.

Outputs:
  results_prxq/collision/summary.csv          long-form one row per (model, split)
  results_prxq/collision/pooled_means.csv     wide-form pooled across splits
  results_prxq/collision/pooled_means.md      markdown of pooled
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

ROOT = Path("results_prxq/collision")


def load_analytic() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "analytic.csv")
    return df.rename(columns={"mae": "mae_F_e"})


def load_mc() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "mc.csv")
    return df.rename(columns={"mae": "mae_F_e"})


def load_dfe() -> pd.DataFrame:
    rows = []
    for split in ["id_test", "length_ood", "family_ood"]:
        p = ROOT / f"dfe_{split}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["split"] = split
        df["model"] = df["S"].apply(lambda s: f"DFE_S={int(s)}")
        df = df[["model", "split", "mae_F_e"]]
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_nn() -> pd.DataFrame:
    rows = []
    for f in sorted(ROOT.glob("*.csv")):
        if f.name.startswith(("analytic", "mc", "dfe_", "summary", "pooled")):
            continue
        df = pd.read_csv(f)
        # eval.py writes one row per (split, length); aggregate to per-(split, model)
        if "model" not in df.columns:
            df["model"] = f.stem.split("_seed")[0]
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    if "mae" in df.columns and "mae_F_e" not in df.columns:
        df = df.rename(columns={"mae": "mae_F_e"})
    keep = [c for c in ["model", "seed", "split", "length", "mae_F_e"] if c in df.columns]
    return df[keep]


def main() -> None:
    parts = []
    for fn, tag in [(load_analytic, "analytic"), (load_mc, "mc"),
                    (load_dfe, "dfe"), (load_nn, "nn")]:
        try:
            d = fn()
            if len(d) > 0:
                d["family"] = tag
                parts.append(d)
        except Exception as e:
            print(f"[warn] {tag}: {e}")
    if not parts:
        print("[error] nothing loaded; run scripts/eval_collision.sh first.")
        return
    df = pd.concat(parts, ignore_index=True, sort=False)
    df.to_csv(ROOT / "summary_long.csv", index=False)

    # Pool over (length, seed) for each (model, split)
    agg_cols = [c for c in ["model", "split"] if c in df.columns]
    pooled = df.groupby(agg_cols, dropna=False)["mae_F_e"].agg(["mean", "std", "count"]).reset_index()
    pooled = pooled.sort_values(["split", "mean"]).reset_index(drop=True)
    pooled.to_csv(ROOT / "pooled_means.csv", index=False)

    print("=== Pooled MAE F_e (collision dataset, lower is better) ===")
    for split in pooled.split.unique():
        sub = pooled[pooled.split == split].sort_values("mean")
        print(f"\n-- split: {split} --")
        for r in sub.itertuples():
            print(f"  {r.model:<28s}  {r.mean:.5f} +- {r.std if r.std==r.std else 0.0:.5f}  (n={int(r.count)})")

    # MD
    lines = ["# PRXQ P1.1 — non-Markovian collision dataset (pooled MAE F_e)", ""]
    for split in pooled.split.unique():
        sub = pooled[pooled.split == split].sort_values("mean")
        lines.append(f"## split: `{split}`\n")
        lines.append("| Model | MAE F_e | std | n |")
        lines.append("|---|---|---|---|")
        for r in sub.itertuples():
            std = r.std if r.std == r.std else 0.0
            lines.append(f"| `{r.model}` | {r.mean:.5f} | {std:.5f} | {int(r.count)} |")
        lines.append("")
    (ROOT / "pooled_means.md").write_text("\n".join(lines))
    print(f"\n[saved] {ROOT/'pooled_means.csv'}")
    print(f"[saved] {ROOT/'pooled_means.md'}")


if __name__ == "__main__":
    main()
