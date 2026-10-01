"""Rebuild revision vector figures and numeric tables from saved server results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('results_revision')
OUT=Path('revision/manuscript')
FIG=OUT/'figures'
FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':13,
    'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':10,
    'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#667085','axes.labelcolor':'#273444','text.color':'#273444',
    'xtick.color':'#475467','ytick.color':'#475467','axes.grid':True,'grid.alpha':.17,
    'grid.linewidth':.6,'lines.linewidth':2.1,'savefig.facecolor':'white'})
C={'dfe':'#475467','neural':'#14998D','ridge':'#DD785A','exchange':'#6876B9','constant':'#B09548'}

def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=.12)
    fig.savefig(FIG/(name+'.png'),dpi=170,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)


fig,axes=plt.subplots(1,2,figsize=(7.4,3.8),layout='constrained')
for family,label,color in [('zz','Dephasing',C['neural']),('exchange','Exchange',C['exchange'])]:
    df=pd.read_csv(ROOT/'stage2'/family/'grid/summary.csv')
    axes[0].plot(np.arange(4),df.empirical_span,marker='o',color=color,label=label)
    continuous=pd.read_csv(ROOT/'stage2'/family/'grid/distribution_sensitivity.csv').uniform_continuous_quadrature_risk.mean()
    risk=[df.iloc[0].discrete_uniform_bayes_mae,df.iloc[-1].discrete_uniform_bayes_mae,continuous]
    axes[1].plot(range(3),risk,marker='o',color=color,linestyle='none')
axes[0].set(xticks=range(4),xticklabels=['15','29','57','113'],xlabel='Retention grid points',ylabel='Mean empirical span',ylim=(.06,.26),title='(a) Span refinement')
axes[0].legend(loc='center right',frameon=False)
axes[1].set(xticks=range(3),xticklabels=['15-point','113-point','Continuous\n(approx.)'],ylabel='Conditional-median MAE',ylim=(.015,.065),title='(b) Distribution-specific risk')
save(fig,'revision_audit')

fig,axes=plt.subplots(1,2,figsize=(7.4,4.6),layout='constrained')
for ax,family,title in zip(axes,['zz','exchange'],['(a) Dephasing','(b) Exchange']):
    df=pd.read_csv(ROOT/'stage2'/family/'validation/summary.csv')
    allocation='zz_known' if family=='zz' else 'fixed'
    for method,label,color,ls in [('dfe','DFE',C['dfe'],'-'),('neural_eb','Neural shrinkage',C['neural'],'-'),
                                 ('summary_ridge_eb','Ridge shrinkage',C['ridge'],'-'),
                                 ('constant_mean_eb','Constant shrinkage',C['constant'],'-')]:
        sub=df[(df.method==method)&(df.allocation==allocation)&(df.shots>=12)&(df.shots<=96)]
        ax.plot(sub.shots,sub.mae,color=color,ls=ls,label=label)
    if family=='exchange':
        val=df[df.method=='summary_ridge_prior'].mae.iloc[0]
        ax.axhline(val,color=C['ridge'],ls=':',lw=2,label='Ridge prior: 0 query shots')
    for target in [.065,.053]:
        ax.axhline(target,color='#AEB6C2',ls='--',lw=.9,zorder=0)
    ax.set(xlabel='Query shots',ylabel='Validation MAE',title=title,xlim=(12,96),ylim=(.025,.15),xticks=[16,32,48,64,80,96])
handles,labels=axes[0].get_legend_handles_labels()
h,l=axes[1].get_legend_handles_labels()
fig.legend(handles+[h[-1]],labels+[l[-1]],loc='outside lower center',ncol=2,frameon=False)
save(fig,'revision_cost_accuracy')

fig,axes=plt.subplots(1,2,figsize=(7.4,3.9),layout='constrained')
queries=np.arange(1,801)
for ax,b,h,c,target in zip(axes,[32,48],[20,36],[32,44],[.065,.053]):
    ax.plot(queries,b*queries/1000,color=C['dfe'],label=f'DFE: {b} shots/query')
    ax.plot(queries,(4096+c*queries)/1000,color=C['constant'],label=f'Constant: {c} shots/query')
    ax.plot(queries,(4096+h*queries)/1000,color=C['neural'],label=f'Ridge / neural: {h} shots/query')
    ax.axvline(4096/12,color='#AEB6C2',ls=':',lw=1.2)
    ax.set(xlabel='Deployment queries',ylabel='Total shots (thousands)',title=f'MAE target {target:.3f}',xlim=(0,800),xticks=[0,200,400,600,800])
    ax.legend(frameon=False,loc='upper left',fontsize=9)
    ax.text(.97,.04,'Strict savings from query 342',transform=ax.transAxes,ha='right',fontsize=9,color='#475467')
save(fig,'revision_amortisation')

fig,axes=plt.subplots(1,2,figsize=(7.4,3.9),layout='constrained')
for ax,family,title in zip(axes,['zz','exchange'],['(a) Dephasing','(b) Exchange']):
    df=pd.read_csv(ROOT/'stage3'/family/'raw_coverage.csv')
    for _,sub in df.groupby('checkpoint'):
        ax.plot(sub.level,sub.coverage,color=C['neural'],alpha=.55,lw=1.5)
    ax.plot([0,1],[0,1],color=C['dfe'],ls='--',lw=1.3,label='Nominal coverage')
    ax.set(xlabel='Quantile level',ylabel='Observed coverage',title=title,xlim=(0,1),ylim=(-.02,1.04),xticks=[0,.2,.4,.6,.8,1])
    ax.legend(loc='lower right',frameon=False,fontsize=9)
save(fig,'revision_uncertainty')

fig,axes=plt.subplots(1,2,figsize=(7.4,4.5),layout='constrained')
for ax,family,title in zip(axes,['zz','exchange'],['(a) Dephasing','(b) Exchange']):
    df=pd.read_csv(ROOT/'stage3'/family/'readout_summary.csv')
    configs=[('dfe',32,'DFE, 32',C['dfe'],'-'),('neural_eb',20,'Neural shrinkage, 20',C['neural'],'-'),
             ('summary_ridge_eb',20,'Ridge shrinkage, 20',C['ridge'],'-')] if family=='zz' else [
             ('dfe',32,'DFE, 32',C['dfe'],'-'),('neural_fusion',32,'Neural linear, 32',C['neural'],'-'),
             ('summary_ridge_fusion',32,'Ridge linear, 32',C['ridge'],'-'),('summary_ridge_prior',0,'Ridge prior, 0',C['ridge'],'--')]
    for method,b,label,color,ls in configs:
        sub=df[(df.method==method)&(df.shots==b)]
        ax.plot(100*sub.readout_error,sub['mean'],marker='o',color=color,ls=ls,label=label)
    ax.set(xlabel='Known readout flip rate (%)',ylabel='Test MAE',title=title,xticks=[0,1,3,5])
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),frameon=False,fontsize=9)
save(fig,'revision_readout')

# Tables are generated from result CSVs rather than manually transcribed.
lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Family / grid & Span & Half-span & Discrete risk\\',r'\midrule']
for family,label in [('zz','Dephasing'),('exchange','Exchange')]:
    for r in pd.read_csv(ROOT/'stage2'/family/'grid/summary.csv').itertuples():
        lines.append(f'{label}, {r.grid_n} & {r.empirical_span:.8f} & {r.worst_case_lower_bound:.8f} & {r.discrete_uniform_bayes_mae:.8f}'+r'\\')
lines += [r'\bottomrule',r'\end{tabular}']
# Four columns; use a matching specification to avoid an empty trailing column.
lines[0]=r'\begin{tabular}{lrrr}'
(OUT/'grid_table.tex').write_text('\n'.join(lines))

lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Prior & ZZ linear & ZZ shrinkage & Exchange linear & Exchange shrinkage\\',r'\midrule']
names=[('constant_mean','Constant mean'),('constant_median','Constant median'),('product_affine','Product'),('exact_marginal_affine','Marginal composition'),('summary_ridge','Ridge'),('neural','Neural')]
for name,label in names:
    values=[]
    for family in ['zz','exchange']:
        df=pd.read_csv(ROOT/'stage2'/family/'test/summary.csv')
        alloc='zz_known' if family=='zz' else 'fixed'
        for suffix in ['_fusion','_eb']:
            selected=df[(df.method==name+suffix)&(df.allocation==alloc)&(df.shots==32)]
            values.append(f'{selected.mae.iloc[0]:.5f}' if len(selected) else '--')
    lines.append(label+' & '+' & '.join(values)+r'\\')
lines += [r'\bottomrule',r'\end{tabular}']
(OUT/'control_table.tex').write_text('\n'.join(lines))

lines=[r'\begin{tabular}{llrrr}',r'\toprule',r'Family & Method & Shots & Clean MAE & 5\% flip MAE\\',r'\midrule']
for family,label in [('zz','ZZ'),('exchange','Exchange')]:
    df=pd.read_csv(ROOT/'stage3'/family/'readout_summary.csv')
    for (method,b),sub in df.groupby(['method','shots']):
        display={'dfe':'DFE','neural_eb':'Neural shrinkage','neural_fusion':'Neural linear',
                 'summary_ridge_eb':'Ridge shrinkage','summary_ridge_fusion':'Ridge linear',
                 'neural_prior':'Neural prior','summary_ridge_prior':'Ridge prior'}[method]
        clean=sub[sub.readout_error==0]['mean'].iloc[0]; noisy=sub[sub.readout_error==.05]['mean'].iloc[0]
        lines.append(f'{label} & {display} & {b} & {clean:.5f} & {noisy:.5f}'+r'\\')
lines += [r'\bottomrule',r'\end{tabular}']
(OUT/'readout_table.tex').write_text('\n'.join(lines))
print('Saved five vector figures and three result-derived tables.')

# Follow-up: validation-selected target curve, without test budget retuning.
fig,axes=plt.subplots(1,2,figsize=(7.4,4.2),layout='constrained')
for ax,family,title in zip(axes,['zz','exchange'],['(a) Dephasing','(b) Exchange']):
    df=pd.read_csv(ROOT/'stage4'/family/'selected_results.csv')
    base=df[(df.role=='dfe')&df.selected].set_index('target')
    for role,label in [('constant','Constant'),('ridge','Ridge'),('neural','Neural')]:
        sub=df[(df.role==role)&df.selected].set_index('target').join(base[['shots','mean_risk_target_pass']],rsuffix='_dfe',how='inner')
        saving=100*(1-sub.shots/sub.shots_dfe)
        ax.plot(sub.index,saving,color=C[role],label=label,lw=1.7,alpha=.8)
        ok=sub.mean_risk_target_pass & sub.mean_risk_target_pass_dfe
        ax.scatter(sub.index[ok],saving[ok],color=C[role],s=26)
        ax.scatter(sub.index[~ok],saving[~ok],facecolors='white',edgecolors=C[role],s=32,zorder=4)
    ax.set(title=title,xlabel='Target MAE',ylabel='Selected query-shot reduction (%)',xticks=[.03,.04,.05,.06,.07,.08],ylim=(-5,105))
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='outside lower center',frameon=False)
save(fig,'revision_target_sweep')

fig,axes=plt.subplots(1,2,figsize=(7.4,4.2),layout='constrained')
df=pd.read_csv(ROOT/'stage4/two_bath/validation/summary.csv')
for m,role,label in [('dfe','dfe','DFE'),('constant_mean_eb','constant','Constant shrinkage'),('summary_ridge_eb','ridge','Ridge shrinkage')]:
    sub=df[df.method==m].sort_values('shots');axes[0].plot(sub.shots,sub.mae,color=C[role],label=label)
prior=df[df.method=='summary_ridge_prior'].mae.iloc[0]
axes[0].axhline(prior,color=C['ridge'],ls=':',label='Ridge prior: 0 shots')
axes[0].set(xlabel='Query shots',ylabel='Validation MAE',title='(a) Two-qubit bath',xlim=(4,256))
audit=pd.read_csv(ROOT/'stage4/two_bath/audit.csv')
axes[1].hist(audit.span113,bins=20,color=C['exchange'],alpha=.85,edgecolor='white')
axes[1].set(xlabel='Fixed-input span',ylabel='Number of sequences',title='(b) Memory ambiguity')
h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2,frameon=False)
save(fig,'revision_two_bath')

labels={'dfe':'DFE','constant':'Constant','ridge':'Ridge','neural':'Neural'}
lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Target & Prior & Shots & MAE & 95th percentile & Pass fraction\\',r'\midrule']
choice=pd.read_csv(ROOT/'stage4/zz/selected_results.csv');dist=pd.read_csv(ROOT/'stage4/zz/deployment_variability.csv')
for target in [.065,.053]:
    for r in choice[choice.target==target].itertuples():
        d=dist[(dist.target==target)&(dist.method==r.method)&(dist.allocation==r.allocation)&(dist.shots==r.shots)].iloc[0]
        lines.append(f'{target:.3f} & {labels[r.role]} & {int(r.shots)} & {r.mae:.5f} & {d.p95:.5f} & {100*d.pass_fraction:.1f}\\%'+r'\\')
lines += [r'\bottomrule',r'\end{tabular}'];(OUT/'deployment_table.tex').write_text('\n'.join(lines))
for family in ['exchange','two_bath']:
    df=pd.read_csv(ROOT/'stage4'/family/'selected_results.csv')
    lines=[r'\begin{tabular}{llrrr}',r'\toprule',r'Target & Method & Shots & Test MAE & Upper 95\%\\',r'\midrule']
    for target in [.065,.053]:
        for r in df[df.target==target].itertuples():
            lines.append(f'{target:.3f} & {labels[r.role]} & {int(r.shots)} & {r.mae:.5f} & {r.upper95:.5f}'+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}'];(OUT/(family+'_followup_table.tex')).write_text('\n'.join(lines))
print('Saved follow-up figures and tables.')
