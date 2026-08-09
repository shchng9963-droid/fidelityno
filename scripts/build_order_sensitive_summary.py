from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / 'data' / 'benchmarks' / 'two_qubit_order_sensitive'
RESULT_DIR = ROOT / 'results' / 'benchmarks' / 'two_qubit_order_sensitive'
OUT_DIR = ROOT / 'results' / 'order_sensitive'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def summarize_gap(split_name: str) -> dict:
    data = np.load(BENCH_DIR / f'{split_name}.npz', allow_pickle=True)
    random_gap = data['perm_gap_random'].astype(float)
    reverse_gap = data['perm_gap_reverse'].astype(float)
    length = data['length'].astype(int)
    return {
        'split': split_name,
        'n': int(len(random_gap)),
        'random_gap_mean': float(random_gap.mean()),
        'random_gap_median': float(np.median(random_gap)),
        'random_gap_p90': float(np.quantile(random_gap, 0.9)),
        'random_gap_frac_gt_001': float(np.mean(random_gap > 0.01)),
        'random_gap_frac_gt_005': float(np.mean(random_gap > 0.05)),
        'reverse_gap_mean': float(reverse_gap.mean()),
        'reverse_gap_median': float(np.median(reverse_gap)),
        'max_length': int(length.max()),
        'min_length': int(length.min()),
    }


def summarize_models() -> tuple[pd.DataFrame, dict]:
    summary = pd.read_csv(RESULT_DIR / 'summary.csv')
    keep_models = ['fidelityno', 'gnn', 'mlp', 'deepsets', 'bidir', 'generic_gnn']
    summary = summary[summary['model'].isin(keep_models)].copy()
    agg = summary.groupby(['model', 'split'])['mae'].agg(['mean', 'std']).reset_index()
    headline = {}
    for split in ['id_test', 'length_ood', 'family_ood']:
        sub = agg[agg['split'] == split].sort_values('mean')
        headline[split] = {
            'best_model': str(sub.iloc[0]['model']),
            'best_mae': float(sub.iloc[0]['mean']),
            'deepsets_mae': float(sub[sub['model'] == 'deepsets']['mean'].iloc[0]),
            'fidelityno_mae': float(sub[sub['model'] == 'fidelityno']['mean'].iloc[0]),
            'deepsets_over_fidelityno': float(sub[sub['model'] == 'deepsets']['mean'].iloc[0] / sub[sub['model'] == 'fidelityno']['mean'].iloc[0]),
        }
    return agg, headline


def write_markdown(gaps: list[dict], agg: pd.DataFrame, headline: dict):
    lines = ['# Two-qubit order-sensitive benchmark summary', '']
    lines.append('## Benchmark difficulty')
    lines.append('')
    lines.append('| Split | n | Lengths | Mean random perm gap | Median random perm gap | P90 random perm gap | Frac > 0.05 | Mean reverse gap |')
    lines.append('|---|---:|---|---:|---:|---:|---:|---:|')
    for row in gaps:
        lines.append(
            f"| {row['split']} | {row['n']} | {row['min_length']}-{row['max_length']} | {row['random_gap_mean']:.4f} | {row['random_gap_median']:.4f} | {row['random_gap_p90']:.4f} | {row['random_gap_frac_gt_005']:.3f} | {row['reverse_gap_mean']:.4f} |"
        )
    lines.append('')
    lines.append('## Model MAE (mean ± std over seeds)')
    lines.append('')
    lines.append('| Model | ID | Length-OOD | Family-OOD |')
    lines.append('|---|---:|---:|---:|')
    order = ['fidelityno', 'gnn', 'generic_gnn', 'bidir', 'mlp', 'deepsets']
    for model in order:
        cells = []
        for split in ['id_test', 'length_ood', 'family_ood']:
            row = agg[(agg['model'] == model) & (agg['split'] == split)]
            if len(row) == 0:
                cells.append('--')
            else:
                cells.append(f"{float(row['mean'].iloc[0]):.4f} ± {float(row['std'].iloc[0]):.4f}")
        lines.append(f"| {model} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append('')
    lines.append('## Headline ratios')
    lines.append('')
    for split, info in headline.items():
        lines.append(
            f"- {split}: best={info['best_model']} ({info['best_mae']:.4f}); DeepSets/FidelityNO-T MAE ratio = {info['deepsets_over_fidelityno']:.2f}x"
        )
    (OUT_DIR / 'summary.md').write_text('\n'.join(lines))


def main():
    gaps = [summarize_gap(split) for split in ['train', 'id_test', 'length_ood', 'family_ood'] if (BENCH_DIR / f'{split}.npz').exists()]
    agg, headline = summarize_models()
    pd.DataFrame(gaps).to_csv(OUT_DIR / 'gap_summary.csv', index=False)
    agg.to_csv(OUT_DIR / 'model_summary.csv', index=False)
    (OUT_DIR / 'headline.json').write_text(json.dumps(headline, indent=2))
    write_markdown(gaps, agg, headline)
    print('wrote', OUT_DIR)


if __name__ == '__main__':
    main()
