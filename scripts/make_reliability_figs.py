#!/usr/bin/env python
"""Reliability + sharpness figures for FidelityNO.

Loads checkpoints (5 seeds) for {fidelityno_large, gnn, mlp, bidir, deepsets},
runs predictions on id_test, length_ood (split by length), family_ood;
applies conformal offsets fit on a held-out fraction of id_test;
produces:
  results/figs/reliability_id.{pdf,png}
  results/figs/reliability_len48.{pdf,png}
  results/figs/sharpness_coverage.{pdf,png}

Reliability = empirical coverage at nominal levels (calibrated curves).
Sharpness = mean predicted IQR width (q90 - q10).
"""
from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Subset
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import make_model
from scripts.eval_calibrated import (
    load_npz, predict, conformal_offsets, apply_offsets,
)

MODELS = {
    'FidelityNO (5M)': 'fidelityno_large',
    'FidelityNO (1M)': 'fidelityno',
    'Generic-GNN':     'gnn',
    'Bidir Trans.':    'bidir',
    'Flat MLP':        'mlp',
    'DeepSets':        'deepsets',
}
SEEDS = [0, 1, 2, 3, 4]
DATA = ROOT / 'data'
CKPT = ROOT / 'checkpoints'
OUT  = ROOT / 'results' / 'figs'
OUT.mkdir(parents=True, exist_ok=True)


def gather_predictions(model_name, target_length=None):
    """Returns {split: (q_calibrated, y, levels)} averaged across seeds.

    target_length: if not None, restrict length_ood to this length.
    """
    bundle = {'id': [], 'len_ood': [], 'fam_ood': []}
    levels_ref = None
    for seed in SEEDS:
        ck_path = CKPT / f'{model_name}_seed{seed}.pt'
        if not ck_path.exists():
            print(f'skip missing {ck_path}'); continue
        ck = torch.load(ck_path, map_location='cpu')
        cfg = OmegaConf.create(ck['cfg'])
        levels = np.asarray(cfg.model.quantiles, dtype=float)
        levels_ref = levels
        raw_id, ds_id = load_npz(DATA / 'id_test.npz')
        input_dim = raw_id['x'].shape[-1]
        max_len = raw_id['x'].shape[1]
        model = make_model(cfg.model.name, input_dim, max_len, cfg)
        model.load_state_dict(ck['model']); model.eval()

        # conformal split inside id_test
        n = len(ds_id)
        rng = np.random.default_rng(123 + seed)
        perm = rng.permutation(n)
        n_cal = n // 2
        cal_idx, eval_idx = perm[:n_cal], perm[n_cal:]
        q_cal, y_cal, _ = predict(model, Subset(ds_id, cal_idx))
        offsets = conformal_offsets(q_cal, y_cal, levels)

        # ID eval
        q_id, y_id, _ = predict(model, Subset(ds_id, eval_idx))
        bundle['id'].append((apply_offsets(q_id, offsets), y_id))

        # length OOD (filter to target_length if requested)
        raw_l, ds_l = load_npz(DATA / 'length_ood.npz')
        q_l, y_l, _ = predict(model, ds_l)
        q_l = apply_offsets(q_l, offsets)
        if target_length is not None:
            mask = raw_l['length'] == target_length
            q_l, y_l = q_l[mask], y_l[mask]
        bundle['len_ood'].append((q_l, y_l))

        # family OOD
        _raw_f, ds_f = load_npz(DATA / 'family_ood.npz')
        q_f, y_f, _ = predict(model, ds_f)
        q_f = apply_offsets(q_f, offsets)
        bundle['fam_ood'].append((q_f, y_f))
    return bundle, levels_ref


def empirical_coverage(q, y, levels):
    return (y[:, None] <= q).mean(0)


COLORS = {
    'FidelityNO (5M)': '#1f77b4',
    'FidelityNO (1M)': '#aec7e8',
    'Generic-GNN':     '#2ca02c',
    'Bidir Trans.':    '#9467bd',
    'Flat MLP':        '#ff7f0e',
    'DeepSets':        '#d62728',
}


def plot_reliability(bundles_by_model, levels, split_key, title, out_stub):
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Ideal', alpha=0.6)
    for label, bundle in bundles_by_model.items():
        if not bundle[split_key]:
            continue
        cov_seeds = []
        for q, y in bundle[split_key]:
            cov_seeds.append(empirical_coverage(q, y, levels))
        cov = np.mean(cov_seeds, axis=0)
        std = np.std(cov_seeds, axis=0)
        ax.plot(levels, cov, '-o', ms=3.5, lw=1.2, color=COLORS[label], label=label)
        ax.fill_between(levels, cov - std, cov + std, color=COLORS[label], alpha=0.15, lw=0)
    ax.set_xlabel('Nominal quantile level')
    ax.set_ylabel('Empirical coverage')
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.legend(fontsize=7, loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f'{out_stub}.pdf')
    fig.savefig(OUT / f'{out_stub}.png', dpi=160)
    plt.close(fig)
    print(f'wrote {out_stub}.pdf')


def plot_sharpness_coverage(bundles_by_model, levels, out_stub):
    """For each (model, split), plot mean IQR width vs |coverage_at_50 - 0.5|.
    Sharper (lower width) + closer to ideal coverage = better.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, key, title in zip(axes,
                               ['id', 'len_ood', 'fam_ood'],
                               ['ID', 'Length-OOD ($n=48$)', 'Family-OOD']):
        for label, bundle in bundles_by_model.items():
            if not bundle[key]:
                continue
            widths, errs = [], []
            for q, y in bundle[key]:
                # 80% interval width: q90 - q10  (levels typically 0.1..0.9)
                i_lo = int(np.argmin(np.abs(levels - 0.1)))
                i_hi = int(np.argmin(np.abs(levels - 0.9)))
                w = (q[:, i_hi] - q[:, i_lo]).mean()
                cov = empirical_coverage(q, y, levels)
                ece = float(np.abs(cov - levels).mean())
                widths.append(w); errs.append(ece)
            w_m, w_s = np.mean(widths), np.std(widths)
            e_m, e_s = np.mean(errs), np.std(errs)
            ax.errorbar(e_m, w_m, xerr=e_s, yerr=w_s,
                        fmt='o', ms=6, color=COLORS[label], label=label,
                        capsize=2, lw=1.0)
        ax.set_xlabel('Calibration error (ECE)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, lw=0.5)
    axes[0].set_ylabel('Sharpness: mean 80% interval width')
    axes[-1].legend(fontsize=7, loc='upper right', frameon=False, bbox_to_anchor=(1.5, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / f'{out_stub}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{out_stub}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_stub}.pdf')


def main():
    bundles = {}
    levels = None
    for label, name in MODELS.items():
        print(f'== {label} ({name}) ==')
        b, lv = gather_predictions(name, target_length=48)
        bundles[label] = b
        if lv is not None:
            levels = lv
    if levels is None:
        print('no checkpoints found, abort'); return

    plot_reliability(bundles, levels, 'id',
                     'Reliability: ID test (calibrated)',
                     'reliability_id')
    plot_reliability(bundles, levels, 'len_ood',
                     'Reliability: length-OOD $n=48$ (calibrated)',
                     'reliability_len48')
    plot_reliability(bundles, levels, 'fam_ood',
                     'Reliability: family-OOD (calibrated)',
                     'reliability_famood')
    plot_sharpness_coverage(bundles, levels, 'sharpness_coverage')


if __name__ == '__main__':
    main()
