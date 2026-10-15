"""Build final-rule readout figure and table from saved server summaries."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('results_revision/final_readout_20261015')
OUT = Path('revision/manuscript')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
    'axes.titlesize': 13, 'axes.labelsize': 12, 'xtick.labelsize': 11,
    'ytick.labelsize': 11, 'legend.fontsize': 9.5, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#667085', 'axes.labelcolor': '#273444',
    'text.color': '#273444', 'xtick.color': '#475467', 'ytick.color': '#475467',
    'axes.grid': True, 'grid.alpha': .17, 'grid.linewidth': .6,
    'lines.linewidth': 2.1, 'savefig.facecolor': 'white'})
styles = [('dfe', 'DFE', '#475467'), ('constant_mean_iv', 'Constant', '#B09548'),
          ('summary_ridge_iv', 'Ridge', '#DD785A'), ('neural_iv', 'Neural', '#14998D')]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.2), layout='constrained')
rows = []
for ax, family, title in zip(axes, ['zz', 'exchange'], ['(a) Dephasing', '(b) Exchange']):
    df = pd.read_csv(ROOT / family / 'summary.csv')
    fixed = {'dfe': 48, 'constant_mean_iv': 44, 'summary_ridge_iv': 36, 'neural_iv': 36} if family == 'zz' else {
        'dfe': 92, 'constant_mean_iv': 72, 'summary_ridge_iv': 3, 'neural_iv': 36}
    for method, label, colour in styles:
        sub = df[(df.method == method) & (df.shots == fixed[method])].sort_values('readout_error')
        assert len(sub) == 4
        ax.plot(100 * sub.readout_error, sub['mean'], 'o-', color=colour, markersize=4,
                label=f'{label}: {fixed[method]} shots')
    ax.set(title=title, xlabel='Known sign-flip rate (%)', ylabel='Test MAE',
           xticks=[0, 1, 3, 5], ylim=(.045, .082))
    ax.legend(frameon=False, loc='upper left')
    methods = styles + [('constant_mean_prior', 'Constant prior', ''),
                       ('summary_ridge_prior', 'Ridge prior', ''), ('neural_prior', 'Neural prior', '')]
    for method, label, _ in methods:
        sub = df[df.method == method]
        for (allocation, shots), config in sub.groupby(['allocation', 'shots'], sort=True):
            clean = config[config.readout_error == 0].iloc[0]['mean']
            noisy = config[config.readout_error == .05].iloc[0]['mean']
            shown = label if 'prior' in label or method == 'dfe' else label + ' shrinkage'
            rows.append(' & '.join(['Dephasing' if family == 'zz' else 'Exchange', shown,
                                  str(int(shots)), f'{clean:.5f}', f'{noisy:.5f}']) + r'\\')
fig.savefig(OUT / 'figures/revision_readout.pdf', bbox_inches='tight', pad_inches=.12)
fig.savefig(OUT / 'figures/revision_readout.png', dpi=170, bbox_inches='tight', pad_inches=.12)
plt.close(fig)
table = '\\begin{tabular}{llrrr}\n\\toprule\nFamily & Method & Shots & Clean MAE & 5\\% flip MAE\\\\\n\\midrule\n'
table += '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n'
(OUT / 'readout_table.tex').write_text(table)
print('Saved final-rule readout figure and table,', len(rows), 'configurations')
