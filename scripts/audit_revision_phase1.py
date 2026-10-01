"""Compact phase-one report; all scientific computation runs on the server."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def main():
    root = Path("results_revision/stage1_pilot")
    df = pd.read_csv(root / "summary.csv")
    df["method"] = df.method.str.replace(r"neural_bidir_seed[0-4]", "neural", regex=True)
    result = df.groupby(["method", "allocation", "shots"], sort=False).mae.mean().reset_index()
    result.to_csv(root / "checkpoint_averaged_summary.csv", index=False)
    chosen = result[result.method.isin(["dfe", "summary_ridge_fusion", "neural_fusion", "neural_eb"])
                    & result.shots.isin([16,32,48,64,96]) & (result.allocation != "pilot")]
    print(chosen.to_string(index=False))
    theta = .01
    lengths = np.array([2,4,8,16])
    eps = np.sin(theta/2)**2
    coherent = pd.DataFrame(dict(length=lengths, step_infidelity=eps,
                                 true_fidelity=np.cos(lengths*theta/2)**2,
                                 product=(1-eps)**lengths,
                                 old_additive_proxy=1-lengths*eps))
    coherent.to_csv(root / "coherent_counterexample.csv", index=False)
    print("Coherent counterexample")
    print(coherent.to_string(index=False))
    print("Manifest")
    print((root / "manifest.json").read_text())


if __name__ == "__main__":
    main()
