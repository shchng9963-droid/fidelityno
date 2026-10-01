"""Development audit of corrected DFE, simple priors, and cross-fitted fusion.

This script uses the previously inspected dataset and must not be reported as
an independent confirmatory experiment. Exact labels are evaluation-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from physics.baselines.hybrid import batch_pauli_expectations_from_choi, fit_convex_fusion
from physics.baselines.revision_measurement import sample_measurement, residual_prior_variance, empirical_bayes_fusion
from scripts.eval_exact_composition import exact_predictions
from scripts.eval_label_budget_baselines import product_predictions, summary_features, fit_ridge, apply_ridge
from scripts.eval_recalibrated import predict_quantiles, fit_recalibrate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budgets", default="4,8,16,24,32,48,64,96,128,256")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--cal-draws", type=int, default=5)
    ap.add_argument("--n-test", type=int, default=2048)
    ap.add_argument("--allocations", default="fixed,pilot,zz_known")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = np.load(args.data, allow_pickle=True)
    y = raw["y"].astype(float)
    ex = batch_pauli_expectations_from_choi(raw["true_choi_real"] + 1j * raw["true_choi_imag"])
    assert np.max(np.abs(ex.mean(axis=1) - y)) < 5e-6
    product, exact = product_predictions(raw), exact_predictions(raw)
    features = summary_features(raw, product, exact)
    sources = {"product_affine": product, "exact_marginal_affine": exact}
    prediction_cache = {}
    for ckpt in args.ckpts:
        print("predict", ckpt, flush=True)
        quantiles, returned = predict_quantiles(ckpt, args.data, args.device)
        np.testing.assert_allclose(returned, y, atol=1e-6)
        key = "neural_" + Path(ckpt).stem
        sources[key] = quantiles.mean(axis=1)
        prediction_cache[key] = quantiles
    if prediction_cache:
        np.savez_compressed(out / "quantiles.npz", **prediction_cache)
    perm = np.random.default_rng(20261001).permutation(len(y))
    test = perm[:args.n_test]
    pool = perm[args.n_test:]
    if len(pool) < 64 * args.cal_draws:
        raise ValueError("not enough separate calibration examples")
    budgets = [int(b) for b in args.budgets.split(",")]
    allocations = args.allocations.split(",")
    rows, error_arrays, ids = [], [], []

    def add(name, allocation, budget, draw, rep, pred, cost, weight=np.nan):
        err = np.clip(pred, 0, 1) - y[test]
        rows.append(dict(method=name, allocation=allocation, shots=budget, cal_draw=draw,
                         repeat=rep, offline_shots=cost, weight=weight, mae=float(np.abs(err).mean()),
                         rmse=float(np.sqrt(np.mean(err**2))), bias=float(err.mean())))
        error_arrays.append(np.abs(err).astype(np.float32))

    def fit_prior(name, training, labels, target_indices):
        if name == "constant_mean":
            return np.full(len(target_indices), np.mean(labels))
        if name == "constant_median":
            return np.full(len(target_indices), np.median(labels))
        if name == "summary_ridge":
            return np.clip(apply_ridge(features[target_indices], fit_ridge(features[training], labels, 1.)), 0, 1)
        slope, intercept = fit_recalibrate(sources[name][training], labels)
        return np.clip(slope * sources[name][target_indices] + intercept, 0, 1)

    # Keep non-recalibrated point estimates beside the adapted results.
    for name, values in sources.items():
        add(name + "_raw", "none", 0, -1, -1, values[test], 0)
    for draw in range(args.cal_draws):
        cal = pool[64*draw:64*(draw+1)]
        ids.append(cal)
        for rep in range(args.repeats):
            label_rng = np.random.default_rng(np.random.SeedSequence([20261001, draw, rep, 1]))
            labels, label_var, _ = sample_measurement(ex[cal], 64, label_rng)
            priors = {}
            for name in ["constant_mean", "constant_median", "summary_ridge", *sources]:
                oof = np.zeros(64)
                for fold in range(4):
                    held = np.arange(64) % 4 == fold
                    oof[held] = fit_prior(name, cal[~held], labels[~held], cal[held])
                prior = fit_prior(name, cal, labels, test)
                priors[name] = (prior, oof, residual_prior_variance(oof, labels, label_var))
                add(name + "_prior", "none", 0, draw, rep, prior, 4096)
            for aidx, allocation in enumerate(allocations):
                for budget in budgets:
                    crng = np.random.default_rng(np.random.SeedSequence([20261001, draw, rep, aidx, budget, 2]))
                    trng = np.random.default_rng(np.random.SeedSequence([20261001, draw, rep, aidx, budget, 3]))
                    cal_obs, _, _ = sample_measurement(ex[cal], budget, crng, allocation)
                    test_obs, test_var, counted = sample_measurement(ex[test], budget, trng, allocation)
                    assert np.all(counted == budget)
                    add("dfe", allocation, budget, draw, rep, test_obs, 0)
                    for name, (prior, oof, tau2) in priors.items():
                        weight = fit_convex_fusion(oof, cal_obs, labels)
                        fused = prior + weight * (test_obs - prior)
                        add(name + "_fusion", allocation, budget, draw, rep, fused, 4096 + 64*budget, weight)
                        eb = empirical_bayes_fusion(prior, test_obs, test_var, tau2)
                        add(name + "_eb", allocation, budget, draw, rep, eb, 4096)
            print("completed calibration draw", draw, "repeat", rep, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    summary = df.groupby(["method", "allocation", "shots"], sort=False).agg(
        mae=("mae", "mean"), run_sd=("mae", "std"), rmse=("rmse", "mean"),
        bias=("bias", "mean"), offline_shots=("offline_shots", "max"), runs=("mae", "size")).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    np.savez_compressed(out / "paired_errors.npz", absolute_errors=np.array(error_arrays), test_indices=test,
                        calibration_indices=np.array(ids), test_truth=y[test], lengths=raw["length"][test])
    manifest = {"status": "development pilot, no matched-accuracy or savings claim",
                "args": vars(args), "elapsed_seconds": time.time() - start,
                "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                "protocol_sha256": hashlib.sha256(Path("revision/statistical_protocol.json").read_bytes()).hexdigest(),
                "checkpoint_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.ckpts},
                "pauli_label_max_error": float(np.max(np.abs(ex.mean(axis=1)-y))),
                "max_abs_chi_z_minus_one": float(np.max(np.abs(ex[:,3]-1))),
                "aggregation": "run SD is descriptive, not a confidence interval; per-query errors retained"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary[summary.shots.isin([16,32,64,96])].to_string(index=False), flush=True)
    print("outputs", out, "seconds", time.time()-start, flush=True)


if __name__ == "__main__":
    main()
