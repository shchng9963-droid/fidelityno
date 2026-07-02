"""Generate cached datasets for the non-Markovian collision-model channel
family (PRXQ P1.1).

Produces splits compatible with v1 train.py / eval.py:
    train.npz, calib.npz, id_test.npz, length_ood.npz, family_ood.npz

Family-OOD is implemented by holding out a different eta-band
(distribution shift on the bath retention strength) rather than a
different functional family — this is the natural notion of
distribution shift for collision models.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm

from physics.channels.collision_nonmarkov import collision_sequence
from physics.composition import composed_stats, sequence_features
from physics.fidelity import (
    FIDELITY_KIND,
    fidelity_formula,
)
from physics.representations import feature_dim_for_representation


def build_split(
    out_path: Path,
    *,
    n: int,
    seed: int,
    lengths: list[int],
    eta_range: tuple[float, float],
    max_len: int,
    representation: str,
    dim: int = 2,
) -> dict:
    rng = np.random.default_rng(seed)
    feat_dim = feature_dim_for_representation(dim, representation)
    X = np.zeros((n, max_len, feat_dim), dtype=np.float32)
    M = np.zeros((n, max_len), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    stats = np.zeros((n, 2), dtype=np.float32)
    lens = np.zeros(n, dtype=np.int32)
    per_fid = np.ones((n, max_len), dtype=np.float32)
    eta_arr = np.zeros(n, dtype=np.float32)
    fam_id = np.empty(n, dtype=object)
    # store the true (non-Markovian) Choi so QI-native baselines like DFE
    # can be evaluated against the *true* dynamics, not the surrogate's
    # marginal view.  shape: (n, d^2 * d^2) real + imag flattened.
    N = dim * dim
    true_choi_real = np.zeros((n, N, N), dtype=np.float32)
    true_choi_imag = np.zeros((n, N, N), dtype=np.float32)

    for i in tqdm(range(n), desc=out_path.name):
        L = int(rng.choice(lengths))
        eta = float(rng.uniform(*eta_range))
        sample = collision_sequence(num_collisions=L, eta=eta, rng=rng)
        x, m = sequence_features(sample.marginals, max_len, dim, representation)
        X[i] = x
        M[i] = m
        y[i] = sample.true_F_e
        true_choi_real[i] = sample.true_choi.real.astype(np.float32)
        true_choi_imag[i] = sample.true_choi.imag.astype(np.float32)
        s = composed_stats(sample.marginals)
        stats[i] = (s["trace"], s["purity"])
        lens[i] = L
        eta_arr[i] = eta
        fam_id[i] = "collision"
        # Per-step F_e to identity, for product-bound and per_fid analyses.
        for t, ch in enumerate(sample.marginals[:max_len]):
            from physics.fidelity import entanglement_fidelity
            per_fid[i, t] = entanglement_fidelity(ch)

    np.savez_compressed(
        out_path,
        x=X,
        mask=M,
        y=y,
        stats=stats,
        length=lens,
        per_fid=per_fid,
        eta=eta_arr,
        true_choi_real=true_choi_real,
        true_choi_imag=true_choi_imag,
        family_prefix=fam_id,
        family_counts=np.zeros((n, 1), dtype=np.int32),
        family_idx_seq=np.full((n, max_len), 0, dtype=np.int16),
        family_names=np.array(["collision"], dtype=object),
        perm_gap_random=np.zeros(n, dtype=np.float32),
        perm_gap_reverse=np.zeros(n, dtype=np.float32),
        fidelity_random_perm=y.copy(),
        fidelity_reverse=y.copy(),
    )
    return {
        "file": str(out_path),
        "n": n,
        "seed": seed,
        "lengths": lengths,
        "eta_range": list(eta_range),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "max_len": max_len,
        "representation": representation,
        "dim": dim,
        "fidelity_kind": FIDELITY_KIND,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="data/collision")
    ap.add_argument("--n-train", type=int, default=80_000)
    ap.add_argument("--n-calib", type=int, default=8_000)
    ap.add_argument("--n-test", type=int, default=8_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--train-lengths", default="2,4,8,16")
    ap.add_argument("--id-lengths", default="2,4,8,16")
    ap.add_argument("--length-ood-lengths", default="24,32,48")
    ap.add_argument("--family-ood-lengths", default="8,16,24")
    ap.add_argument("--train-eta", default="0.0,0.7",
                    help="eta range for train+id+length-OOD")
    ap.add_argument("--ood-eta", default="0.85,0.99",
                    help="eta range held out for family-OOD test")
    ap.add_argument("--representation", default="choi_hermitian")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    parse_lens = lambda s: sorted({int(x) for x in s.split(",") if x.strip()})
    parse_eta = lambda s: tuple(float(x) for x in s.split(","))

    train_lens = parse_lens(args.train_lengths)
    id_lens = parse_lens(args.id_lengths)
    L_ood = parse_lens(args.length_ood_lengths)
    F_ood = parse_lens(args.family_ood_lengths)
    eta_train = parse_eta(args.train_eta)
    eta_ood = parse_eta(args.ood_eta)

    splits = []
    splits.append(build_split(outdir / "train.npz", n=args.n_train, seed=args.seed,
                              lengths=train_lens, eta_range=eta_train,
                              max_len=args.max_len, representation=args.representation))
    splits.append(build_split(outdir / "calib.npz", n=args.n_calib, seed=args.seed + 100,
                              lengths=id_lens, eta_range=eta_train,
                              max_len=args.max_len, representation=args.representation))
    splits.append(build_split(outdir / "id_test.npz", n=args.n_test, seed=args.seed + 1,
                              lengths=id_lens, eta_range=eta_train,
                              max_len=args.max_len, representation=args.representation))
    splits.append(build_split(outdir / "length_ood.npz", n=args.n_test, seed=args.seed + 2,
                              lengths=L_ood, eta_range=eta_train,
                              max_len=args.max_len, representation=args.representation))
    splits.append(build_split(outdir / "family_ood.npz", n=args.n_test, seed=args.seed + 3,
                              lengths=F_ood, eta_range=eta_ood,
                              max_len=args.max_len, representation=args.representation))

    manifest = {
        "regime": "non_markovian_collision",
        "fidelity_kind": FIDELITY_KIND,
        "fidelity_formula": fidelity_formula(),
        "splits": splits,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[saved] {outdir}/manifest.json")
    for s in splits:
        print(f"  {Path(s['file']).name:14s} n={s['n']:6d} y_mean={s['y_mean']:.4f} eta={s['eta_range']}")


if __name__ == "__main__":
    main()
