from __future__ import annotations
import argparse, time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from physics.channels.two_qubit import cnot_unitary, swap_unitary

LEVELS = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], dtype=np.float64)


def infer_dim_from_feature_dim(feature_dim: int) -> int:
    # feature_dim = 2 * (d^2 x d^2 complex Choi entries)
    d = int(round((feature_dim / 2) ** 0.25))
    if 2 * (d * d) ** 2 != feature_dim:
        raise ValueError(f"cannot infer Hilbert dimension from feature_dim={feature_dim}")
    return d


def features_to_choi(features: np.ndarray) -> np.ndarray:
    f = np.asarray(features)
    if f.ndim != 1:
        raise ValueError("features_to_choi expects one flat feature vector")
    half = f.size // 2
    d = infer_dim_from_feature_dim(f.size)
    n = d * d
    real = f[:half].reshape(n, n)
    imag = f[half:].reshape(n, n)
    choi = real + 1j * imag
    return 0.5 * (choi + choi.conj().T)


def kraus_from_choi(choi: np.ndarray, dim: int, atol: float = 1e-10) -> list[np.ndarray]:
    herm = 0.5 * (choi + choi.conj().T)
    vals, vecs = np.linalg.eigh(herm)
    kraus = []
    for val, v in zip(vals, vecs.T):
        if val > atol:
            kraus.append(np.sqrt(float(val)) * v.reshape((dim, dim), order="F"))
    return kraus


def exact_process_fidelity_from_kraus(kraus_sequence: list[list[np.ndarray]], dim: int, target_unitary: np.ndarray | None = None) -> float:
    """Exact entanglement fidelity F_e = (1/d^2) sum_i |Tr(U^dagger K_i)|^2.

    Computed by enumerating all Kraus paths of the composed channel.
    Equivalent to ``physics.composition.process_fidelity`` (which uses
    Choi inner product) up to floating-point error.
    """
    target_unitary = np.eye(dim, dtype=np.complex128) if target_unitary is None else np.asarray(target_unitary, dtype=np.complex128)
    paths = [np.eye(dim, dtype=np.complex128)]
    for kraus in kraus_sequence:
        paths = [k @ p for p in paths for k in kraus]
    val = sum(abs(np.trace(target_unitary.conj().T @ k)) ** 2 for k in paths) / (dim ** 2)
    return float(np.clip(np.real(val), 0.0, 1.0))


def mc_process_fidelity(kraus_sequence: list[list[np.ndarray]], dim: int, samples: int, rng: np.random.Generator, target_unitary: np.ndarray | None = None) -> float:
    """Monte-Carlo estimator of entanglement fidelity F_e via Kraus-path importance sampling.

    Proposal q_i(k) = ||K||_F^2 / d (valid for TP maps); estimator is
    unbiased for F_e and has Var = O(1/samples). The function name keeps
    "process_fidelity" for back-compat with v1; the quantity is F_e in
    the convention of ``physics.fidelity``.
    """
    target_unitary = np.eye(dim, dtype=np.complex128) if target_unitary is None else np.asarray(target_unitary, dtype=np.complex128)
    proposals = []
    for kraus in kraus_sequence:
        probs = np.array([np.linalg.norm(k, "fro") ** 2 / dim for k in kraus], dtype=np.float64)
        probs = np.clip(probs, 0, None)
        probs = probs / probs.sum()
        proposals.append(probs)
    acc = 0.0
    for _ in range(samples):
        kprod = np.eye(dim, dtype=np.complex128)
        q = 1.0
        for kraus, probs in zip(kraus_sequence, proposals):
            idx = int(rng.choice(len(kraus), p=probs))
            q *= float(probs[idx])
            kprod = kraus[idx] @ kprod
        acc += (abs(np.trace(target_unitary.conj().T @ kprod)) ** 2 / (dim ** 2)) / max(q, 1e-300)
    return float(np.clip(acc / samples, 0.0, 1.0))


def target_unitary_from_family_indices(indices: np.ndarray, family_names: np.ndarray, dim: int) -> np.ndarray:
    unitary = np.eye(dim, dtype=np.complex128)
    names = family_names.tolist() if isinstance(family_names, np.ndarray) else list(family_names)
    for idx in indices:
        if idx < 0:
            break
        family = names[int(idx)]
        if family == 'imperfect_cnot':
            step = cnot_unitary()
        elif family == 'imperfect_swap':
            step = swap_unitary()
        else:
            step = np.eye(dim, dtype=np.complex128)
        unitary = step @ unitary
    return unitary


def quantile_rows_from_point(pred: np.ndarray, y: np.ndarray, length: np.ndarray, split: str, model: str, seed: int, latency_ms: float) -> list[dict]:
    q = np.repeat(pred[:, None], len(LEVELS), axis=1)
    cov = (y[:, None] <= q).mean(axis=0)
    ece = float(np.abs(cov - LEVELS).mean())
    e = y[:, None] - q
    pinball = float(np.maximum(LEVELS[None, :] * e, (LEVELS[None, :] - 1) * e).mean())
    rows = []
    for L in sorted(set(length.tolist())):
        idx = length == L
        rows.append({
            "model": model,
            "seed": seed,
            "split": split,
            "length": int(L),
            "mae": float(np.abs(pred[idx] - y[idx]).mean()),
            "pinball": pinball,
            "crps": 2 * pinball,
            "ece": ece,
            "latency_ms": latency_ms,
        })
    return rows


def evaluate_split(path: Path, split: str, budgets: list[int], seed: int, max_eval: int | None) -> list[dict]:
    data = np.load(path, allow_pickle=True)
    x, mask, y, length = data["x"], data["mask"], data["y"], data["length"]
    family_idx_seq = data["family_idx_seq"] if "family_idx_seq" in data.files else None
    family_names = data["family_names"] if "family_names" in data.files else None
    if max_eval is not None:
        x, mask, y, length = x[:max_eval], mask[:max_eval], y[:max_eval], length[:max_eval]
        if family_idx_seq is not None:
            family_idx_seq = family_idx_seq[:max_eval]
    dim = infer_dim_from_feature_dim(x.shape[-1])
    seqs = []
    target_unitaries = []
    for i in range(len(y)):
        seqs.append([kraus_from_choi(features_to_choi(x[i, j]), dim) for j in range(int(mask[i].sum()))])
        if family_idx_seq is not None and family_names is not None:
            target_unitaries.append(target_unitary_from_family_indices(family_idx_seq[i], family_names, dim))
        else:
            target_unitaries.append(np.eye(dim, dtype=np.complex128))
    rows = []
    for budget in budgets:
        rng = np.random.default_rng(seed + 1009 * budget)
        preds = np.zeros(len(y), dtype=np.float64)
        t0 = time.perf_counter()
        for i, seq in enumerate(seqs):
            preds[i] = mc_process_fidelity(seq, dim, budget, rng, target_unitary=target_unitaries[i])
        latency_ms = 1000 * (time.perf_counter() - t0) / max(1, len(y))
        rows += quantile_rows_from_point(preds, y, length, split, f"mc_{budget}", seed, latency_ms)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results/mc.csv")
    ap.add_argument("--budgets", default="10,100,1000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=None, help="optional cap for smoke tests")
    args = ap.parse_args()
    budgets = [int(x) for x in args.budgets.split(",") if x]
    rows = []
    for split in ["id_test", "length_ood", "family_ood"]:
        rows += evaluate_split(Path(args.data_dir) / f"{split}.npz", split, budgets, args.seed, args.max_eval)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
