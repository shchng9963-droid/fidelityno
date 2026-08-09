from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

BENCHMARKS = [
    ("single_qubit_mixed", 2, "mixed", "pauli"),
    ("single_qubit_lindblad_holdout", 2, "mixed", "lindblad"),
    ("two_qubit_mixed", 4, "mixed", "correlated_dephasing"),
    ("two_qubit_order_sensitive", 4, "order_sensitive", "correlated_dephasing"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="data/benchmarks")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--representation", default="choi_hermitian", choices=["choi_hermitian", "raw_choi_flat", "ptm"])
    args = ap.parse_args()
    root = Path(args.out_root); root.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    for name, dim, family, holdout in BENCHMARKS:
        out = root / name
        extra = []
        if family == "order_sensitive":
            extra = ["--train-lengths", "8,16", "--id-lengths", "8,16", "--length-ood-lengths", "24,32,48", "--family-ood-lengths", "8,16,24"]
        cmd = [py, "scripts/gen_data.py", "--outdir", str(out), "--n-train", str(args.n_train), "--n-test", str(args.n_test), "--seed", str(args.seed), "--dim", str(dim), "--family", family, "--holdout-family", holdout, "--representation", args.representation] + extra
        print("running", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
