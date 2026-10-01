"""Exploratory paired intervals with crossed checkpoints, draws and queries."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results_revision/stage1_pilot")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    root = Path(args.root)
    runs = pd.read_csv(root / "runs.csv")
    archive = np.load(root / "paired_errors.npz")
    errors = archive["absolute_errors"]
    assert len(runs) == len(errors)
    index = {(r.method, r.allocation, int(r.shots), int(r.cal_draw), int(r.repeat)): i
             for i, r in enumerate(runs.itertuples())}
    draws = sorted(runs[runs.cal_draw >= 0].cal_draw.unique())
    repeats = sorted(runs[runs.repeat >= 0].repeat.unique())
    neural = sorted(set(runs[runs.method.str.match(r"neural_bidir_seed[0-4]_fusion$")].method))
    results = []
    for allocation in ["fixed", "zz_known"]:
        for other in ["dfe", "summary_ridge_fusion"]:
            # Shape: calibration draw, measurement repeat, checkpoint, query.
            # Baseline measurements are shared across checkpoints, not copied
            # and treated as independently observed data.
            diff = np.array([[[errors[index[(m, allocation, 32, d, r)]]
                              - errors[index[(other, allocation, 32, d, r)]]
                              for m in neural] for r in repeats] for d in draws], dtype=float)
            nd, nr, ns, nq = diff.shape
            rng = np.random.default_rng(202610017)
            values = np.empty(args.bootstrap)
            for b in range(args.bootstrap):
                chosen_d = rng.integers(nd, size=nd)
                chosen_s = rng.integers(ns, size=ns)
                chosen_q = rng.integers(nq, size=nq)
                curve = np.zeros(nq)
                for d in chosen_d:
                    chosen_r = rng.integers(nr, size=nr)
                    curve += diff[d][chosen_r][:, chosen_s].mean(axis=(0,1)) / nd
                values[b] = curve[chosen_q].mean()
            low, high = np.quantile(values, [.025, .975])
            results.append(dict(allocation=allocation, shots=32, contrast="neural_fusion minus " + other,
                                mae_difference=diff.mean(), ci95_low=low, ci95_high=high,
                                bootstrap=args.bootstrap, interpretation="exploratory, previously inspected dataset"))
    result = pd.DataFrame(results)
    result.to_csv(root / "exploratory_paired_intervals.csv", index=False)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
