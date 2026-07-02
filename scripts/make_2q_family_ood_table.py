"""C3: Generate 2-qubit family-OOD diagnostic table for the paper.

Hold-out family: correlated_dephasing.
Active train families: two_qubit_depolarizing, imperfect_cnot, imperfect_swap.

Outputs:
  - results/two_qubit_family_ood_table.csv  (paper-ready table)
  - results/two_qubit_family_ood_table.tex  (LaTeX booktabs)
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

SUMMARY = 'results/benchmarks/two_qubit_mixed/summary.csv'
DATA_DIR = Path('data/benchmarks/two_qubit_mixed')

df = pd.read_csv(SUMMARY)
fam = df[df['split'] == 'family_ood'].copy()

# Per-length structure of the held-out family
ystats = []
d = np.load(DATA_DIR / 'family_ood.npz')
for L in sorted(set(d['length'].tolist())):
    idx = d['length'] == L
    y = d['y'][idx]
    ystats.append({'length': int(L), 'n': int(idx.sum()),
                   'y_mean': float(y.mean()), 'y_std': float(y.std()),
                   'mae_constant_ymean': float(np.abs(y.mean() - y).mean())})
ystats_df = pd.DataFrame(ystats)
print('=== Held-out family target statistics (correlated_dephasing) ===')
print(ystats_df.to_string(index=False))
print()

# Aggregate MAE per (model, length) over seeds
g = fam.groupby(['model', 'length'])['mae'].agg(['mean', 'std']).reset_index()
piv_mean = g.pivot(index='model', columns='length', values='mean')
piv_std = g.pivot(index='model', columns='length', values='std')

# Order
order = ['product_bound', 'fvg_bound', 'mc_10', 'mc_100', 'mlp', 'gnn', 'fidelityno']
piv_mean = piv_mean.reindex(order)
piv_std = piv_std.reindex(order)

print('=== 2-qubit Family-OOD MAE (mean over 5 seeds; held-out: correlated_dephasing) ===')
print(piv_mean.round(4))

# Add the constant-predictor reference (per-length oracle baseline)
const_baseline = {L: ystats_df[ystats_df['length'] == L]['mae_constant_ymean'].values[0]
                  for L in piv_mean.columns}
piv_mean.loc['constant(y_mean)'] = [const_baseline[L] for L in piv_mean.columns]
piv_std.loc['constant(y_mean)'] = [0.0] * len(piv_mean.columns)

# Save CSV
out_csv = Path('results/two_qubit_family_ood_table.csv')
out_csv.parent.mkdir(parents=True, exist_ok=True)
combined = piv_mean.copy()
for L in piv_mean.columns:
    combined[f'{L}_std'] = piv_std[L]
combined.to_csv(out_csv)
print(f'\nwrote {out_csv}')

# LaTeX
def fmt(m, s):
    if pd.isna(m):
        return '--'
    if pd.isna(s) or s == 0:
        return f'{m:.3f}'
    return f'{m:.3f}\\,$\\pm$\\,{s:.3f}'

display = {
    'product_bound': 'Product Bound (B1)',
    'fvg_bound': 'FvG Bound (B2)',
    'mc_10': 'MC-10 (B3)',
    'mc_100': 'MC-100 (B3)',
    'mlp': 'Flat MLP (B4)',
    'gnn': 'Linear-chain GNN (B7)',
    'fidelityno': '\\textbf{FidelityNO (ours)}',
    'constant(y_mean)': '\\emph{const.\\ $\\bar y$ (oracle)}',
}

lines = [
    r'\begin{table}[t]',
    r'\centering',
    r'\caption{Two-qubit family-OOD diagnostic. Models trained on \{depolarizing, imperfect CNOT, imperfect SWAP\}; tested on the held-out \emph{correlated dephasing} family. MAE on fidelity prediction (mean $\pm$ std over 5 seeds). The held-out target distribution is highly concentrated (std $\\approx$ 0.02--0.06 within each length), so a constant predictor at the held-out mean is a strong reference.}',
    r'\label{tab:two_qubit_family_ood}',
    r'\begin{tabular}{lccc}',
    r'\toprule',
    r'Model & $n{=}8$ & $n{=}16$ & $n{=}24$ \\',
    r'\midrule',
]
for m in piv_mean.index:
    row = display.get(m, m)
    cells = [fmt(piv_mean.loc[m, L], piv_std.loc[m, L]) for L in [8, 16, 24]]
    lines.append(f'{row} & ' + ' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

out_tex = Path('results/two_qubit_family_ood_table.tex')
out_tex.write_text('\n'.join(lines))
print(f'wrote {out_tex}')

# Diagnostic note
diag = f"""# Two-qubit family-OOD diagnostic (held-out: correlated_dephasing)

## Target distribution is near-constant within each length

| length | n | y_mean | y_std | MAE@const.(y_mean) |
|--------|---|--------|-------|--------------------|
""" + '\n'.join(
    f"| {r['length']} | {r['n']} | {r['y_mean']:.4f} | {r['y_std']:.4f} | {r['mae_constant_ymean']:.4f} |"
    for r in ystats
) + f"""

## Interpretation

The held-out family `correlated_dephasing`, when composed with itself for n hops,
produces fidelities concentrated within a band of width `std <= 0.06` per length.
A constant predictor at the held-out marginal mean already achieves
MAE in [{min(r['mae_constant_ymean'] for r in ystats):.3f},
       {max(r['mae_constant_ymean'] for r in ystats):.3f}].

Consequences for honest reporting:
  1. The "best" model on this split is the one closest to that constant predictor,
     i.e. Flat MLP (B4), which has effectively collapsed to predicting the
     training-y mean — this is NOT genuine generalization.
  2. FidelityNO performs WORSE than B1 on this split because its causal sequence
     model EXTRAPOLATES the unfamiliar parameter direction, producing predictions
     that are correctly structured (length-decreasing) but mis-scaled.
  3. This is a real negative result. The 2Q benchmark held-out family is not
     a useful generalization probe: it has insufficient target variance.

## Recommendation for paper

Report this table in the appendix as a diagnostic. In the main text, retain the
1-qubit family-OOD result (broader y-range) as the family-OOD headline, and
present the 2-qubit family-OOD only with the "constant(y_mean)" oracle row to
contextualize that the metric is degenerate here.
"""
Path('results/two_qubit_family_ood_diagnostic.md').write_text(diag)
print('wrote results/two_qubit_family_ood_diagnostic.md')
