"""Generate one collision split with eta on an independent RNG stream.

Using independent streams makes the intended statistical independence between
the hidden bath-retention coefficient and observable collision parameters
explicit and prevents accidental pseudo-random-state leakage across fields.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm

from physics.channels.collision_nonmarkov import collision_sequence
from physics.composition import composed_stats, sequence_features
from physics.fidelity import entanglement_fidelity
from physics.representations import feature_dim_for_representation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--parameter-seed", type=int, default=20260807)
    ap.add_argument("--eta-seed", type=int, default=920260807)
    ap.add_argument("--lengths", default="8,16,24")
    ap.add_argument("--eta-range", default="0.85,0.99")
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--representation", default="choi_hermitian")
    args = ap.parse_args()

    lengths = np.array(sorted({int(value) for value in args.lengths.split(",") if value.strip()}))
    eta_low, eta_high = (float(value) for value in args.eta_range.split(","))
    parameter_rng = np.random.default_rng(args.parameter_seed)
    eta_rng = np.random.default_rng(args.eta_seed)
    feat_dim = feature_dim_for_representation(2, args.representation)

    x = np.zeros((args.n, args.max_len, feat_dim), dtype=np.float32)
    mask = np.zeros((args.n, args.max_len), dtype=np.float32)
    y = np.zeros(args.n, dtype=np.float32)
    stats = np.zeros((args.n, 2), dtype=np.float32)
    sequence_lengths = np.zeros(args.n, dtype=np.int32)
    per_fid = np.ones((args.n, args.max_len), dtype=np.float32)
    eta_values = eta_rng.uniform(eta_low, eta_high, size=args.n).astype(np.float32)
    true_real = np.zeros((args.n, 4, 4), dtype=np.float32)
    true_imag = np.zeros((args.n, 4, 4), dtype=np.float32)

    for index in tqdm(range(args.n), desc=Path(args.out).name):
        length = int(parameter_rng.choice(lengths))
        sample = collision_sequence(
            length,
            eta=float(eta_values[index]),
            rng=parameter_rng,
        )
        features, valid = sequence_features(
            sample.marginals, args.max_len, 2, args.representation
        )
        x[index], mask[index] = features, valid
        y[index] = sample.true_F_e
        summary = composed_stats(sample.marginals)
        stats[index] = (summary["trace"], summary["purity"])
        sequence_lengths[index] = length
        true_real[index] = sample.true_choi.real.astype(np.float32)
        true_imag[index] = sample.true_choi.imag.astype(np.float32)
        for step, channel in enumerate(sample.marginals):
            per_fid[index, step] = entanglement_fidelity(channel)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        x=x,
        mask=mask,
        y=y,
        stats=stats,
        length=sequence_lengths,
        per_fid=per_fid,
        eta=eta_values,
        true_choi_real=true_real,
        true_choi_imag=true_imag,
        parameter_seed=np.array(args.parameter_seed),
        eta_seed=np.array(args.eta_seed),
        rng_streams_independent=np.array(True),
        family_prefix=np.full(args.n, "collision", dtype=object),
        family_counts=np.zeros((args.n, 1), dtype=np.int32),
        family_idx_seq=np.zeros((args.n, args.max_len), dtype=np.int16),
        family_names=np.array(["collision"], dtype=object),
        perm_gap_random=np.zeros(args.n, dtype=np.float32),
        perm_gap_reverse=np.zeros(args.n, dtype=np.float32),
        fidelity_random_perm=y.copy(),
        fidelity_reverse=y.copy(),
    )
    print(
        f"[saved] {out} n={args.n} y_mean={y.mean():.6f} "
        f"y_std={y.std():.6f} eta=[{eta_values.min():.6f}, {eta_values.max():.6f}]"
    )


if __name__ == "__main__":
    main()
