#!/usr/bin/env python3
"""Generate all paper figures from results/summary.csv."""
from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Nature-like style
plt.rcParams.update({
    'font.size': 9,
    'axes.linewidth': 0.6,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'legend.fontsize': 7.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.4,
    'lines.markersize': 4,
})

OUTDIR = Path('results/figs')
OUTDIR.mkdir(parents=True, exist_ok=True)

# Muted color palette
COLORS = {
    'fidelityno': '#2171B5',
    'fidelityno_large': '#084594',
    'bidir': '#6BAED6',
    'mlp': '#E6550D',
    'deepsets': '#FD8D3C',
    'gnn': '#31A354',
    'generic_gnn': '#74C476',
    'product_bound': '#756BB1',
    'fvg_bound': '#9E9AC8',
    'mc_10': '#969696',
    'mc_100': '#636363',
    'mc_1000': '#252525',
}
MARKERS = {
    'fidelityno': 'o',
    'fidelityno_large': 's',
    'bidir': '^',
    'mlp': 'D',
    'deepsets': 'v',
    'gnn': 'p',
    'generic_gnn': 'h',
    'product_bound': 'x',
    'fvg_bound': '+',
    'mc_10': '<',
    'mc_100': '>',
    'mc_1000': '*',
}
LABELS = {
    'fidelityno': 'FidelityNO',
    'fidelityno_large': 'FidelityNO-L',
    'bidir': 'Bidir-Transformer',
    'mlp': 'Flat MLP (B4)',
    'deepsets': 'DeepSets (B5)',
    'gnn': 'GNN (B7)',
    'generic_gnn': 'Generic-GNN',
    'product_bound': 'Product Bound (B1)',
    'fvg_bound': 'FvG Bound (B2)',
    'mc_10': 'MC-10 (B3)',
    'mc_100': 'MC-100 (B3)',
    'mc_1000': 'MC-1000 (B3)',
}


def load_data():
    df = pd.read_csv('results/summary.csv')
    return df


def get_color(m):
    return COLORS.get(m, '#999999')

def get_marker(m):
    return MARKERS.get(m, '.')

def get_label(m):
    return LABELS.get(m, m)


def plot_length_vs_mae(df):
    """Fig 1: MAE vs sequence length for all models, one subplot per split."""
    splits = ['id_test', 'length_ood', 'family_ood']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

    for ax, split in zip(axes, splits):
        sub = df[df['split'] == split]
        for model, g in sub.groupby('model'):
            agg = g.groupby('length')['mae'].agg(['mean', 'std']).reset_index()
            ax.plot(agg['length'], agg['mean'], marker=get_marker(model),
                    color=get_color(model), label=get_label(model))
            if 'std' in agg.columns and agg['std'].notna().any():
                ax.fill_between(agg['length'],
                                agg['mean'] - agg['std'],
                                agg['mean'] + agg['std'],
                                alpha=0.12, color=get_color(model))
        ax.set_xlabel('Sequence length')
        ax.set_title(split.replace('_', ' ').title())
        ax.set_xscale('log', base=2)

    axes[0].set_ylabel('Fidelity MAE')
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'length_vs_mae.pdf', bbox_inches='tight')
    fig.savefig(OUTDIR / 'length_vs_mae.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  length_vs_mae')


def plot_length_generalization(df):
    """Fig 2: Focused length generalization — ID+OOD lengths on one axis."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    # Combine id_test + length_ood for neural models
    neural_models = ['fidelityno', 'fidelityno_large', 'bidir', 'mlp', 'deepsets', 'gnn', 'generic_gnn']
    combined = df[df['model'].isin(neural_models)]
    combined = combined[combined['split'].isin(['id_test', 'length_ood'])]

    for model, g in combined.groupby('model'):
        agg = g.groupby('length')['mae'].agg(['mean', 'std']).reset_index()
        ax.plot(agg['length'], agg['mean'], marker=get_marker(model),
                color=get_color(model), label=get_label(model))
        if agg['std'].notna().any():
            ax.fill_between(agg['length'],
                            agg['mean'] - agg['std'],
                            agg['mean'] + agg['std'],
                            alpha=0.12, color=get_color(model))

    # Add vertical line for train/test boundary
    ax.axvline(x=16, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.text(16, ax.get_ylim()[1]*0.95, 'train max', fontsize=7, ha='right', color='gray')

    ax.set_xlabel('Sequence length')
    ax.set_ylabel('Fidelity MAE')
    ax.set_title('Length Generalization')
    ax.set_xscale('log', base=2)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'length_generalization.pdf', bbox_inches='tight')
    fig.savefig(OUTDIR / 'length_generalization.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  length_generalization')


def plot_calibration(df):
    """Fig 3: ECE bar chart per model (ID split)."""
    id_df = df[df['split'] == 'id_test']
    ece_avg = id_df.groupby('model')['ece'].mean().sort_values()

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = [get_color(m) for m in ece_avg.index]
    labels = [get_label(m) for m in ece_avg.index]
    ax.barh(range(len(ece_avg)), ece_avg.values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(ece_avg)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Expected Calibration Error (ECE)')
    ax.set_title('Calibration (ID Split)')
    ax.axvline(x=0.05, color='red', linestyle='--', alpha=0.5, linewidth=0.8, label='Target ECE ≤ 0.05')
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'calibration_ece.pdf', bbox_inches='tight')
    fig.savefig(OUTDIR / 'calibration_ece.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  calibration_ece')


def plot_crps_comparison(df):
    """Fig 4: CRPS comparison bar chart."""
    id_df = df[df['split'] == 'id_test']
    crps_avg = id_df.groupby('model')['crps'].mean().sort_values()

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = [get_color(m) for m in crps_avg.index]
    labels = [get_label(m) for m in crps_avg.index]
    ax.barh(range(len(crps_avg)), crps_avg.values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(crps_avg)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('CRPS')
    ax.set_title('Distributional Quality (ID Split)')
    fig.tight_layout()
    fig.savefig(OUTDIR / 'crps_comparison.pdf', bbox_inches='tight')
    fig.savefig(OUTDIR / 'crps_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  crps_comparison')


def plot_latency_comparison(df):
    """Fig 5: Latency comparison."""
    id_df = df[df['split'] == 'id_test']
    lat_avg = id_df.groupby('model')['latency_ms'].mean().sort_values()

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = [get_color(m) for m in lat_avg.index]
    labels = [get_label(m) for m in lat_avg.index]
    ax.barh(range(len(lat_avg)), lat_avg.values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(lat_avg)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Latency (ms / sequence)')
    ax.set_xscale('log')
    ax.set_title('Inference Latency')
    fig.tight_layout()
    fig.savefig(OUTDIR / 'latency_comparison.pdf', bbox_inches='tight')
    fig.savefig(OUTDIR / 'latency_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  latency_comparison')


def plot_family_ood(df):
    """Fig 6: Family OOD MAE heatmap / bar chart."""
    fam_df = df[df['split'] == 'family_ood']
    if fam_df.empty:
        print('  (skipping family_ood — no data)')
        return

    mae_avg = fam_df.groupby('model')['mae'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = [get_color(m) for m in mae_avg.index]
    labels = [get_label(m) for m in mae_avg.index]
    ax.barh(range(len(mae_avg)), mae_avg.values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(mae_avg)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Fidelity MAE')
    ax.set_title('Family OOD Generalization')
    fig.tight_layout()
    fig.savefig(OUTDIR / 'family_ood_mae.pdf', bbox_inches='tight')
    fig.savefig(OUTDIR / 'family_ood_mae.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  family_ood_mae')


def make_summary_table(df):
    """Table 1: Summary metrics per model averaged across seeds."""
    metrics = ['mae', 'pinball', 'crps', 'ece']
    rows = []
    for model in df['model'].unique():
        for split in ['id_test', 'length_ood', 'family_ood']:
            sub = df[(df['model'] == model) & (df['split'] == split)]
            if sub.empty:
                continue
            row = {'model': get_label(model), 'split': split}
            for m in metrics:
                if m in sub.columns:
                    row[f'{m}_mean'] = sub[m].mean()
                    row[f'{m}_std'] = sub[m].std()
            if 'latency_ms' in sub.columns:
                row['latency_ms'] = sub['latency_ms'].mean()
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv('results/summary_table.csv', index=False)
    print(f'  Wrote results/summary_table.csv')
    return summary


def main():
    print('Generating figures...')
    df = load_data()
    plot_length_vs_mae(df)
    plot_length_generalization(df)
    plot_calibration(df)
    plot_crps_comparison(df)
    plot_latency_comparison(df)
    plot_family_ood(df)
    make_summary_table(df)
    print(f'Done. Figures in {OUTDIR}/')


if __name__ == '__main__':
    main()
