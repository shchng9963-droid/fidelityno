#!/usr/bin/env python3
"""C2: evaluate length_extreme checkpoints on {id_test_short, id_test_full, length_ood}.

length_extreme = trained on n<=4 only, tested on n in {2,4} (id_short),
n up to 16 (id_full), and n in {24,32,48} (length_ood). The standard eval.py
expects a fixed split-name set; this wrapper handles the custom names and
writes one CSV per checkpoint to results/length_extreme/.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import eval_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='checkpoints/length_extreme')
    ap.add_argument('--data-dir', default='data/length_extreme')
    ap.add_argument('--out-dir', default='results/length_extreme')
    ap.add_argument('--patterns', nargs='+',
                    default=['fidelityno', 'bidir', 'gnn', 'generic_gnn', 'mlp', 'deepsets'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        'id_short':   str(Path(args.data_dir) / 'id_test_short.npz'),
        'id_full':    str(Path(args.data_dir) / 'id_test_full.npz'),
        'length_ood': str(Path(args.data_dir) / 'length_ood.npz'),
    }

    for p in args.patterns:
        for s in args.seeds:
            ckpt = Path(args.ckpt_dir) / f'{p}_seed{s}.pt'
            if not ckpt.exists():
                print(f'  skip missing: {ckpt}'); continue
            out = out_dir / f'{p}_seed{s}.csv'
            print(f'  eval {ckpt} -> {out}')
            eval_ckpt(str(ckpt), splits, str(out))


if __name__ == '__main__':
    main()
