#!/usr/bin/env python3
"""Demo A: Bayesian Optimization of Memory Cutoff using FidelityNO as Surrogate.

Scenario: A quantum repeater chain has n channels. We want to find the optimal
"memory cutoff" parameter c that maximizes end-to-end fidelity. The cutoff c
controls how many channels in the chain are used (simulating a purification
protocol that trades depth for fidelity).

We compare three approaches:
1. Exact grid search: compute ground-truth fidelity for every cutoff (expensive)
2. MC surrogate: use Monte Carlo sampling to approximate fidelity, then optimize
3. FidelityNO surrogate: use the trained neural surrogate (fast)

Report: wall-clock time, quality of found optimum, number of evaluations.
"""
from __future__ import annotations
import argparse, sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from physics.channels.single_qubit import sample_single_qubit
from physics.composition import exact_sequence_fidelity, sequence_features
from scripts.eval_mc import features_to_choi, kraus_from_choi, mc_process_fidelity, infer_dim_from_feature_dim
from train import make_model, mean_from_prediction


def generate_chain(n_channels: int, seed: int, dim: int = 2):
    """Generate a fixed chain of n_channels noisy quantum channels."""
    rng = np.random.default_rng(seed)
    families = ["amplitude_damping", "phase_damping", "depolarizing"]
    channels = []
    for _ in range(n_channels):
        fam = rng.choice(families)
        channels.append(sample_single_qubit(rng, fam))
    return channels


def exact_fidelity_at_cutoff(channels, cutoff):
    """Ground truth: compose first `cutoff` channels, compute exact fidelity."""
    seq = channels[:cutoff]
    return exact_sequence_fidelity(seq)


def mc_fidelity_at_cutoff(channels, cutoff, n_mc: int, dim: int, seed: int):
    """MC estimate of fidelity for first `cutoff` channels."""
    seq = channels[:cutoff]
    max_len = len(channels)
    feats, mask = sequence_features(seq, max_len, dim)
    rng = np.random.default_rng(seed)
    kraus_seq = [kraus_from_choi(features_to_choi(feats[j]), dim) for j in range(cutoff)]
    return mc_process_fidelity(kraus_seq, dim, n_mc, rng)


def neural_fidelity_at_cutoff(model, channels, cutoff, dim: int, max_len: int, device: str):
    """Neural surrogate prediction for first `cutoff` channels."""
    seq = channels[:cutoff]
    feats, mask = sequence_features(seq, max_len, dim)
    x = torch.tensor(feats).float().unsqueeze(0).to(device)
    m = torch.tensor(mask).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred, _ = model(x, m)
        return float(mean_from_prediction(pred).cpu().item())


def bo_with_surrogate(eval_fn, cutoffs, n_init: int = 5, n_iter: int = 15, seed: int = 42):
    """Simple BO: random init, then greedy pick best from evaluated + exploration.

    Uses UCB-like strategy: pick cutoff with highest (predicted + uncertainty bonus).
    Since we don't have a GP, use simple epsilon-greedy with shrinking epsilon.
    """
    rng = np.random.default_rng(seed)
    evaluated = {}

    # Random initialization
    init_cutoffs = rng.choice(cutoffs, size=min(n_init, len(cutoffs)), replace=False)
    for c in init_cutoffs:
        evaluated[int(c)] = eval_fn(c)

    # Greedy optimization with exploration
    for i in range(n_iter):
        epsilon = max(0.1, 1.0 - i / n_iter)  # decay exploration
        if rng.random() < epsilon:
            # Explore: pick random unevaluated cutoff
            unevaluated = [c for c in cutoffs if c not in evaluated]
            if unevaluated:
                c = int(rng.choice(unevaluated))
            else:
                break
        else:
            # Exploit: evaluate neighbor of current best
            best_c = max(evaluated, key=evaluated.get)
            neighbors = [c for c in cutoffs if abs(c - best_c) <= 3 and c not in evaluated]
            if neighbors:
                c = int(rng.choice(neighbors))
            else:
                unevaluated = [c for c in cutoffs if c not in evaluated]
                if unevaluated:
                    c = int(rng.choice(unevaluated))
                else:
                    break
        evaluated[int(c)] = eval_fn(c)

    best_c = max(evaluated, key=evaluated.get)
    return best_c, evaluated[best_c], len(evaluated)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/fidelityno_seed0.pt')
    ap.add_argument('--chain-length', type=int, default=32)
    ap.add_argument('--mc-budget', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='results/demo_cutoff_bo.csv')
    args = ap.parse_args()

    dim = 2
    chain = generate_chain(args.chain_length, seed=args.seed, dim=dim)
    cutoffs = list(range(2, args.chain_length + 1))

    # --- Method 1: Exact Grid Search ---
    print(f"[1/3] Exact grid search over {len(cutoffs)} cutoffs...")
    t0 = time.perf_counter()
    exact_vals = {c: exact_fidelity_at_cutoff(chain, c) for c in cutoffs}
    exact_time = time.perf_counter() - t0
    exact_best_c = max(exact_vals, key=exact_vals.get)
    print(f"  Best cutoff={exact_best_c}, F={exact_vals[exact_best_c]:.6f}, time={exact_time:.3f}s")

    # --- Method 2: MC Surrogate BO ---
    print(f"[2/3] MC-{args.mc_budget} surrogate BO...")
    t0 = time.perf_counter()
    mc_best_c, mc_best_val, mc_evals = bo_with_surrogate(
        lambda c: mc_fidelity_at_cutoff(chain, c, args.mc_budget, dim, args.seed + c),
        cutoffs, n_init=5, n_iter=15, seed=args.seed
    )
    mc_time = time.perf_counter() - t0
    mc_true_val = exact_vals[mc_best_c]
    print(f"  Best cutoff={mc_best_c}, F_pred={mc_best_val:.6f}, F_true={mc_true_val:.6f}, "
          f"evals={mc_evals}, time={mc_time:.3f}s")

    # --- Method 3: Neural Surrogate BO ---
    print(f"[3/3] FidelityNO surrogate BO...")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"  WARNING: {ckpt_path} not found, skipping neural surrogate")
        neural_best_c, neural_best_val, neural_evals, neural_time = -1, -1.0, 0, 0.0
        neural_true_val = -1.0
    else:
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        cfg = OmegaConf.create(ck['cfg'])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        max_len = 48  # from data gen
        input_dim = 32  # single qubit choi_hermitian
        model = make_model(cfg.model.name, input_dim, max_len, cfg).to(device)
        model.load_state_dict(ck['model'])
        model.eval()

        t0 = time.perf_counter()
        neural_best_c, neural_best_val, neural_evals = bo_with_surrogate(
            lambda c: neural_fidelity_at_cutoff(model, chain, c, dim, max_len, device),
            cutoffs, n_init=5, n_iter=15, seed=args.seed
        )
        neural_time = time.perf_counter() - t0
        neural_true_val = exact_vals[neural_best_c]
        print(f"  Best cutoff={neural_best_c}, F_pred={neural_best_val:.6f}, F_true={neural_true_val:.6f}, "
              f"evals={neural_evals}, time={neural_time:.3f}s")

    # --- Report ---
    rows = [
        {
            'method': 'exact_grid',
            'best_cutoff': exact_best_c,
            'best_fidelity': exact_vals[exact_best_c],
            'true_fidelity': exact_vals[exact_best_c],
            'n_evals': len(cutoffs),
            'wall_clock_s': exact_time,
            'speedup_vs_exact': 1.0,
            'regret': 0.0,
        },
        {
            'method': f'mc_{args.mc_budget}_bo',
            'best_cutoff': mc_best_c,
            'best_fidelity': mc_best_val,
            'true_fidelity': mc_true_val,
            'n_evals': mc_evals,
            'wall_clock_s': mc_time,
            'speedup_vs_exact': exact_time / max(mc_time, 1e-9),
            'regret': exact_vals[exact_best_c] - mc_true_val,
        },
    ]
    if neural_best_c > 0:
        rows.append({
            'method': 'fidelityno_bo',
            'best_cutoff': neural_best_c,
            'best_fidelity': neural_best_val,
            'true_fidelity': neural_true_val,
            'n_evals': neural_evals,
            'wall_clock_s': neural_time,
            'speedup_vs_exact': exact_time / max(neural_time, 1e-9),
            'regret': exact_vals[exact_best_c] - neural_true_val,
        })

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nResults written to {args.out}")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
