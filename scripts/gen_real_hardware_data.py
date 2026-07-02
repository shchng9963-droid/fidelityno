"""Generate a real-hardware evaluation dataset for FidelityNO.

For each named IBM Fake*V2 backend, sample N sequences of single-qubit
channels (using the device's actual T1, T2, gate error, gate time per
qubit) at a range of lengths. Compose, compute exact F_e, encode with
v1's `sequence_features`, and save in the .npz format that `eval.py`
already consumes (no model retraining required).

Output layout (one directory per backend):
  data/real_hardware/<backend>/
    real_hw_test.npz       # the only split needed for eval
    manifest.json          # backend metadata + per-qubit calibration
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm

from physics.composition import exact_sequence_fidelity, sequence_features, composed_stats, channel_reference_fidelity
from physics.devices.ibm_backend import (
    available_fake_backends,
    device_qubit_channel,
    list_calibrated_qubits,
    load_fake_backend,
)
from physics.fidelity import FIDELITY_KIND, fidelity_formula


DEFAULT_BACKENDS = (
    "FakeKolkataV2",
    "FakeMumbaiV2",
    "FakeHanoiV2",
    "FakeMontrealV2",
    "FakeCairoV2",
    "FakeWashingtonV2",
)

FAMILIES_1Q = ["amplitude_damping", "phase_damping", "depolarizing", "pauli", "lindblad"]


def build_one_backend(
    backend_name: str,
    out_root: Path,
    n_seq: int,
    lengths: list[int],
    max_len: int,
    representation: str = "choi_hermitian",
    seed: int = 0,
) -> dict:
    """Build a real-hardware test split using one IBM Fake*V2 backend."""
    rng = np.random.default_rng(seed)
    backend = load_fake_backend(backend_name)
    qubits = list_calibrated_qubits(backend)
    if not qubits:
        raise RuntimeError(f"{backend_name} has no calibrated single-qubit gates.")

    # Pre-build the per-qubit Choi cache for this backend.
    qubit_channels: dict[int, np.ndarray] = {}
    qubit_meta: dict[int, dict] = {}
    for q in qubits:
        ch = device_qubit_channel(backend, q)
        qubit_channels[q] = ch
        qubit_meta[q] = {
            "T1_us": float(ch.metadata["T1_us"]),
            "T2_us": float(ch.metadata["T2_us"]),
            "gate_time_us": float(ch.metadata["gate_time_us"]),
            "rb_error": float(ch.metadata["rb_error"]),
            "gamma": float(ch.params[0]),
            "lambda_p": float(ch.params[1]),
            "p_dep": float(ch.params[2]),
        }

    from physics.representations import feature_dim_for_representation
    feat_dim = feature_dim_for_representation(2, representation)

    X = np.zeros((n_seq, max_len, feat_dim), dtype=np.float32)
    mask = np.zeros((n_seq, max_len), dtype=np.float32)
    y = np.zeros(n_seq, dtype=np.float32)
    stats = np.zeros((n_seq, 2), dtype=np.float32)
    lens = np.zeros(n_seq, dtype=np.int32)
    per_fid = np.ones((n_seq, max_len), dtype=np.float32)
    fam_id = np.empty(n_seq, dtype=object)
    family_counts = np.zeros((n_seq, len(FAMILIES_1Q)), dtype=np.int32)
    family_idx_seq = np.full((n_seq, max_len), -1, dtype=np.int16)
    perm_gap_random = np.zeros(n_seq, np.float32)
    perm_gap_reverse = np.zeros(n_seq, np.float32)
    fidelity_random_perm = np.zeros(n_seq, np.float32)
    fidelity_reverse = np.zeros(n_seq, np.float32)

    qubit_array = np.array(qubits)
    for i in tqdm(range(n_seq), desc=f"gen {backend_name}"):
        L = int(rng.choice(lengths))
        chosen = rng.choice(qubit_array, size=L, replace=True)
        seq = [qubit_channels[int(q)] for q in chosen]
        X[i], mask[i] = sequence_features(seq, max_len, 2, representation)
        y[i] = exact_sequence_fidelity(seq)
        st = composed_stats(seq)
        stats[i] = [st["trace"], st["purity"]]
        lens[i] = L
        per_fid[i, :L] = [channel_reference_fidelity(ch) for ch in seq]
        fam_id[i] = ",".join(ch.name for ch in seq[: min(3, L)])
        # Real-hardware sequences are all amplitude_damping ∘ phase_damping ∘
        # depolarizing — we use the AD index for compatibility with v1
        # family-counting code.
        for pos in range(L):
            family_idx_seq[i, pos] = 0
            family_counts[i, 0] += 1

    out_dir = out_root / backend_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "real_hw_test.npz"
    np.savez_compressed(
        out_file,
        x=X, mask=mask, y=y, stats=stats, length=lens,
        per_fid=per_fid, family_prefix=fam_id,
        family_counts=family_counts, family_idx_seq=family_idx_seq,
        family_names=np.array(FAMILIES_1Q, dtype=object),
        perm_gap_random=perm_gap_random,
        perm_gap_reverse=perm_gap_reverse,
        fidelity_random_perm=fidelity_random_perm,
        fidelity_reverse=fidelity_reverse,
    )
    manifest = {
        "backend": backend_name,
        "qiskit_backend_name": backend.name,
        "num_qubits": int(backend.num_qubits),
        "n_seq": int(n_seq),
        "lengths": lengths,
        "max_len": max_len,
        "seed": seed,
        "representation": representation,
        "fidelity_kind": FIDELITY_KIND,
        "fidelity_formula": fidelity_formula(),
        "y_description": (
            "Per-sequence entanglement fidelity F_e for a sequence of single-qubit "
            "device-noise channels (amplitude damping, phase damping, depolarizing) "
            "where parameters are derived from the IBM Fake*V2 calibration snapshot."
        ),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "qubits_used": qubits,
        "per_qubit_calibration": qubit_meta,
        "out_file": str(out_file),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="data/real_hardware")
    ap.add_argument("--backends", nargs="*", default=list(DEFAULT_BACKENDS))
    ap.add_argument("--n-seq", type=int, default=512,
                    help="Sequences per backend (sampled across lengths).")
    ap.add_argument("--lengths", default="2,4,8,16,24,32,48",
                    help="Comma-separated allowed sequence lengths.")
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--representation", default="choi_hermitian",
                    choices=["choi_hermitian", "raw_choi_flat", "compressed_hermitian", "ptm"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="List available fake backends and exit.")
    args = ap.parse_args()

    if args.list:
        for n in available_fake_backends():
            print(n)
        return

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    summary = []
    for name in args.backends:
        t0 = time.perf_counter()
        m = build_one_backend(
            backend_name=name,
            out_root=out_root,
            n_seq=args.n_seq,
            lengths=lengths,
            max_len=args.max_len,
            representation=args.representation,
            seed=args.seed,
        )
        m["wall_time_s"] = time.perf_counter() - t0
        summary.append(m)
        print(
            f"[{name}] n={m['n_seq']} y_mean={m['y_mean']:.4f} y_std={m['y_std']:.4f} "
            f"wall={m['wall_time_s']:.1f}s -> {m['out_file']}"
        )

    (out_root / "manifest.json").write_text(json.dumps({
        "fidelity_kind": FIDELITY_KIND,
        "fidelity_formula": fidelity_formula(),
        "splits": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
