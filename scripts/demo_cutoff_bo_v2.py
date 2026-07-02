#!/usr/bin/env python3
"""B6 / Demo A wall-clock framing.

Honest framing: at n=32 single-qubit channels, exact Choi composition is already
cheap (a few ms), so a per-query neural call cannot win on wall clock alone.
The surrogate's advantage shows up when:

  (1) the per-query exact cost grows (longer chains, larger Hilbert space, or
      Lindblad time-integration), and/or
  (2) the BO loop calls the surrogate many times in a tight inner loop where
      latency-per-call rather than absolute cost dominates.

This script:
  - Runs the cutoff-BO demo across multiple chain lengths {16, 32, 48, 64} and
    seeds {0..4}.
  - Compares exact-grid, MC-100, MC-1000, and FidelityNO surrogate.
  - Reports regret (quality), n_evals, and wall-clock with mean +- std.
  - Computes a fair "amortized speedup": (per-query latency)_exact / (per-query
    latency)_neural averaged across queries inside the BO loop. This isolates
    the surrogate property the paper actually cares about.

Output: results/demos/demo_cutoff_bo_table.csv (raw rows)
        results/demos/demo_cutoff_bo_aggregate.csv (mean +- std)
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from physics.channels.single_qubit import sample_single_qubit
from physics.composition import exact_sequence_fidelity, sequence_features
from scripts.eval_mc import features_to_choi, kraus_from_choi, mc_process_fidelity
from train import make_model, mean_from_prediction


def generate_chain(n_channels: int, seed: int):
    rng = np.random.default_rng(seed)
    fams = ["amplitude_damping", "phase_damping", "depolarizing"]
    return [sample_single_qubit(rng, rng.choice(fams)) for _ in range(n_channels)]


def timed_eval(fn):
    """Wrap an eval_fn so it accumulates per-call latency."""
    latencies = []
    def wrapped(c):
        t0 = time.perf_counter()
        v = fn(c)
        latencies.append(time.perf_counter() - t0)
        return v
    return wrapped, latencies


def bo_loop(eval_fn, cutoffs, n_init=5, n_iter=15, seed=42):
    rng = np.random.default_rng(seed)
    evaluated = {}
    init = rng.choice(cutoffs, size=min(n_init, len(cutoffs)), replace=False)
    for c in init:
        evaluated[int(c)] = eval_fn(int(c))
    for i in range(n_iter):
        eps = max(0.1, 1.0 - i / n_iter)
        if rng.random() < eps:
            unevaluated = [c for c in cutoffs if c not in evaluated]
            if not unevaluated: break
            c = int(rng.choice(unevaluated))
        else:
            best_c = max(evaluated, key=evaluated.get)
            neighbors = [c for c in cutoffs if abs(c - best_c) <= 3 and c not in evaluated]
            if neighbors:
                c = int(rng.choice(neighbors))
            else:
                unevaluated = [c for c in cutoffs if c not in evaluated]
                if not unevaluated: break
                c = int(rng.choice(unevaluated))
        evaluated[c] = eval_fn(c)
    best_c = max(evaluated, key=evaluated.get)
    return best_c, evaluated[best_c], len(evaluated)


def run_one(model, device, chain_length: int, seed: int):
    chain = generate_chain(chain_length, seed=seed)
    cutoffs = list(range(2, chain_length + 1))
    dim = 2
    max_len_pad = 48 if chain_length <= 48 else chain_length

    # --- Exact grid ---
    t0 = time.perf_counter()
    exact_vals = {c: exact_sequence_fidelity(chain[:c]) for c in cutoffs}
    exact_time = time.perf_counter() - t0
    exact_best = max(exact_vals, key=exact_vals.get)
    exact_per_call = exact_time / len(cutoffs)

    rows = [{
        'chain_length': chain_length, 'seed': seed,
        'method': 'exact_grid', 'best_cutoff': exact_best,
        'best_pred': exact_vals[exact_best], 'true_at_pred': exact_vals[exact_best],
        'regret': 0.0, 'n_evals': len(cutoffs),
        'wall_clock_s': exact_time, 'per_call_ms': 1e3 * exact_per_call,
    }]

    # --- MC-100 and MC-1000 ---
    for mc_budget in (100, 1000):
        feats_full, _ = sequence_features(chain, max_len_pad, dim)
        kraus_full = [kraus_from_choi(features_to_choi(feats_full[j]), dim) for j in range(chain_length)]
        def mc_fn(c, _kr=kraus_full, _mc=mc_budget, _seed=seed):
            rng = np.random.default_rng(_seed + c)
            return mc_process_fidelity(_kr[:c], dim, _mc, rng)
        wrapped, lats = timed_eval(mc_fn)
        t0 = time.perf_counter()
        bc, bv, nev = bo_loop(wrapped, cutoffs, seed=seed)
        wall = time.perf_counter() - t0
        rows.append({
            'chain_length': chain_length, 'seed': seed,
            'method': f'mc_{mc_budget}_bo', 'best_cutoff': bc,
            'best_pred': bv, 'true_at_pred': exact_vals[bc],
            'regret': exact_vals[exact_best] - exact_vals[bc],
            'n_evals': nev, 'wall_clock_s': wall,
            'per_call_ms': 1e3 * float(np.mean(lats)),
        })

    # --- FidelityNO surrogate ---
    if model is not None:
        feats_full, mask_full = sequence_features(chain, max_len_pad, dim)
        feats_t = torch.tensor(feats_full).float().to(device)
        def neu_fn(c):
            x = feats_t[:c].unsqueeze(0)
            m = torch.ones(1, c, device=device)
            with torch.no_grad():
                pred, _ = model(x, m)
                return float(mean_from_prediction(pred).cpu().item())
        # warm-up
        _ = neu_fn(cutoffs[0])
        wrapped, lats = timed_eval(neu_fn)
        t0 = time.perf_counter()
        bc, bv, nev = bo_loop(wrapped, cutoffs, seed=seed)
        wall = time.perf_counter() - t0
        rows.append({
            'chain_length': chain_length, 'seed': seed,
            'method': 'fidelityno_bo', 'best_cutoff': bc,
            'best_pred': bv, 'true_at_pred': exact_vals[bc],
            'regret': exact_vals[exact_best] - exact_vals[bc],
            'n_evals': nev, 'wall_clock_s': wall,
            'per_call_ms': 1e3 * float(np.mean(lats)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/fidelityno_seed0.pt')
    ap.add_argument('--lengths', nargs='+', type=int, default=[16, 32, 48])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--out-dir', default='results/demos')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = None
    ckpt_path = Path(args.ckpt)
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        cfg = OmegaConf.create(ck['cfg'])
        max_len = 48
        input_dim = 32
        model = make_model(cfg.model.name, input_dim, max_len, cfg).to(device)
        model.load_state_dict(ck['model'])
        model.eval()
        print(f"loaded {ckpt_path}")
    else:
        print(f"WARN: ckpt {ckpt_path} missing -- skipping neural surrogate")

    all_rows = []
    for L in args.lengths:
        for s in args.seeds:
            print(f"  chain_length={L} seed={s}")
            all_rows.extend(run_one(model, device, L, s))
    df = pd.DataFrame(all_rows)
    raw_path = out_dir / 'demo_cutoff_bo_table.csv'
    df.to_csv(raw_path, index=False)
    print(f"wrote {raw_path}: {len(df)} rows")

    agg = (df.groupby(['chain_length', 'method'], as_index=False)
             .agg(regret_mean=('regret','mean'), regret_std=('regret','std'),
                  wall_clock_mean=('wall_clock_s','mean'),
                  wall_clock_std=('wall_clock_s','std'),
                  per_call_ms_mean=('per_call_ms','mean'),
                  per_call_ms_std=('per_call_ms','std'),
                  n_evals_mean=('n_evals','mean')))
    # amortized per-call speedup vs MC-1000 and vs exact_grid
    pivot = agg.pivot(index='chain_length', columns='method', values='per_call_ms_mean')
    pivot.columns = [f'per_call_ms_{c}' for c in pivot.columns]
    pivot = pivot.reset_index()
    if 'per_call_ms_fidelityno_bo' in pivot.columns:
        if 'per_call_ms_exact_grid' in pivot.columns:
            pivot['speedup_vs_exact_per_call'] = pivot['per_call_ms_exact_grid'] / pivot['per_call_ms_fidelityno_bo']
        if 'per_call_ms_mc_1000_bo' in pivot.columns:
            pivot['speedup_vs_mc1000_per_call'] = pivot['per_call_ms_mc_1000_bo'] / pivot['per_call_ms_fidelityno_bo']
    speed_path = out_dir / 'demo_cutoff_bo_speedup.csv'
    pivot.to_csv(speed_path, index=False)
    agg_path = out_dir / 'demo_cutoff_bo_aggregate.csv'
    agg.to_csv(agg_path, index=False)
    print(f"wrote {agg_path}")
    print(f"wrote {speed_path}")
    print('\nAggregate (regret, mean wall-clock, per-call ms):')
    print(agg.to_string(index=False))
    print('\nPer-call speedups:')
    print(pivot.to_string(index=False))


if __name__ == '__main__':
    main()
