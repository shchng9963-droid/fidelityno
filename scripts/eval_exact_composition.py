"""Benchmark deterministic composition when full per-step Choi matrices are inputs.

This baseline is exact for Markovian channel sequences represented by their full
Choi matrices. On collision-model data it composes the observable reset-bath
marginals, so its error measures hidden-memory mismatch rather than numerical
approximation. The timed path uses direct superoperator multiplication and does
not construct Python ``Channel`` objects.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scripts.eval_mc import target_unitary_from_family_indices


def infer_dim(feature_dim: int) -> int:
    dim = int(round((feature_dim / 2) ** 0.25))
    if 2 * dim**4 != feature_dim:
        raise ValueError(
            "exact composition requires full real+imaginary Choi features; "
            f"got feature_dim={feature_dim}"
        )
    return dim


def features_to_superop(feat: np.ndarray, dim: int) -> np.ndarray:
    """Convert flattened real+imaginary Choi features without Python loops."""
    n = dim * dim
    choi = feat[: n * n].reshape(n, n) + 1j * feat[n * n :].reshape(n, n)
    # C[i,a,j,b] -> S[a+b*d, i+j*d] for column-major vectorization.
    return choi.reshape(dim, dim, dim, dim).transpose(3, 1, 2, 0).reshape(n, n)


def exact_predictions(raw: np.lib.npyio.NpzFile, limit: int | None = None) -> np.ndarray:
    x = raw["x"]
    mask = raw["mask"]
    n_samples = len(x) if limit is None else min(limit, len(x))
    dim = infer_dim(x.shape[-1])
    family_idx = raw["family_idx_seq"] if "family_idx_seq" in raw.files else None
    family_names = raw["family_names"] if "family_names" in raw.files else None
    pred = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        total = np.eye(dim * dim, dtype=np.complex128)
        for j in range(int(mask[i].sum())):
            total = features_to_superop(x[i, j], dim) @ total
        if family_idx is None or family_names is None:
            target_u = np.eye(dim, dtype=np.complex128)
        else:
            target_u = target_unitary_from_family_indices(family_idx[i], family_names, dim)
        target_s = np.kron(target_u, target_u.conj())
        pred[i] = float(np.clip(np.trace(target_s.conj().T @ total).real / dim**2, 0.0, 1.0))
    return pred


def benchmark(path: Path, repeats: int, warmup: int, max_eval: int | None) -> dict:
    raw = np.load(path, allow_pickle=True)
    n = len(raw["y"]) if max_eval is None else min(max_eval, len(raw["y"]))
    for _ in range(warmup):
        exact_predictions(raw, n)
    timings = []
    pred = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        pred = exact_predictions(raw, n)
        timings.append(time.perf_counter() - t0)
    assert pred is not None
    y = raw["y"][:n].astype(np.float64)
    err = np.abs(pred - y)
    return {
        "split": path.stem,
        "path": str(path),
        "n": n,
        "dim": infer_dim(raw["x"].shape[-1]),
        "mae_F_e": float(err.mean()),
        "max_abs_err_F_e": float(err.max()),
        "latency_ms_per_seq_median": 1000.0 * float(np.median(timings)) / n,
        "latency_ms_per_seq_min": 1000.0 * float(np.min(timings)) / n,
        "repeats": repeats,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-eval", type=int)
    ap.add_argument("--out", default="results_mlst/exact_composition.csv")
    args = ap.parse_args()

    rows = [benchmark(Path(p), args.repeats, args.warmup, args.max_eval) for p in args.data]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.8g}"))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
