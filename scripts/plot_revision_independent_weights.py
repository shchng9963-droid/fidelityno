"""Rebuild final-weight figures and tables only from saved server results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('results_revision/stage5');OUT=Path('revision/manuscript');FIG=OUT/'figures'
FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':13,
    'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':10,
    'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#667085','axes.labelcolor':'#273444','text.color':'#273444',
    'xtick.color':'#475467','ytick.color':'#475467','axes.grid':True,'grid.alpha':.17,
    'grid.linewidth':.6,'lines.linewidth':2.1,'savefig.facecolor':'white'})
C={'dfe':'#475467','neural':'#14998D','ridge':'#DD785A','constant':'#B09548'}
METHOD={'dfe':'dfe','constant':'constant_mean_iv','ridge':'summary_ridge_iv','neural':'neural_iv'}
def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=.12)
    fig.savefig(FIG/(name+'.png'),dpi=170,bbox_inches='tight',pad_inches=.12);plt.close(fig)
def selected(family):return pd.read_csv(ROOT/family/'selected_results.csv')
def head(family,target,role):
    d=selected(family);return d[(d.target==target)&(d.role==role)].iloc[0]
def table(name,cols,header,rows):
    text='\\begin{tabular}{'+cols+'}\n\\toprule\n'+' & '.join(header)+r'\\'+'\n\\midrule\n'
    text+='\n'.join(' & '.join(row)+r'\\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n'
    (OUT/name).write_text(text)

fig,axes=plt.subplots(1,2,figsize=(7.4,4.6),layout='constrained')
for ax,family,title in zip(axes,['zz','exchange'],['(a) Dephasing','(b) Exchange']):
    df=pd.read_csv(ROOT/family/'validation/summary.csv');allocation='zz_known' if family=='zz' else 'fixed'
    for role in ['dfe','constant','ridge','neural']:
        sub=df[(df.method==METHOD[role])&(df.allocation==allocation)&(df.shots>=12)&(df.shots<=96)]
        ax.plot(sub.shots,sub.mae,color=C[role],label='DFE' if role=='dfe' else role.title()+' shrinkage')
    if family=='exchange':ax.axhline(df[df.method=='summary_ridge_prior'].mae.iloc[0],color=C['ridge'],ls=':',label='Ridge prior: 0 query shots')
    for target in [.065,.053]:ax.axhline(target,color='#AEB6C2',ls='--',lw=.9,zorder=0)
    ax.set(xlabel='Query shots',ylabel='Validation MAE',title=title,xlim=(12,96),ylim=(.025,.15),xticks=[16,32,48,64,80,96])
h,l=axes[0].get_legend_handles_labels();hh,ll=axes[1].get_legend_handles_labels()
fig.legend(h+[hh[-1]],l+[ll[-1]],loc='outside lower center',ncol=2,frameon=False)
save(fig,'revision_cost_accuracy')

fig,axes=plt.subplots(1,2,figsize=(7.4,3.9),layout='constrained');queries=np.arange(1,801)
for ax,t in zip(axes,[.065,.053]):
    b=int(head('zz',t,'dfe').shots);h=int(head('zz',t,'neural').shots);c=int(head('zz',t,'constant').shots)
    assert h==int(head('zz',t,'ridge').shots)
    ax.plot(queries,b*queries/1000,color=C['dfe'],label=f'DFE: {b} shots/query')
    ax.plot(queries,(4096+c*queries)/1000,color=C['constant'],label=f'Constant: {c} shots/query')
    ax.plot(queries,(4096+h*queries)/1000,color=C['neural'],label=f'Ridge / neural: {h} shots/query')
    crossing=4096/(b-h);ax.axvline(crossing,color='#AEB6C2',ls=':',lw=1.2)
    ax.set(xlabel='Deployment queries',ylabel='Total shots (thousands)',title=f'MAE target {t:.3f}',xlim=(0,800),xticks=[0,200,400,600,800])
    ax.legend(frameon=False,loc='upper left',fontsize=9)
    ax.text(.97,.04,f'Strict savings from query {int(crossing)+1}',transform=ax.transAxes,ha='right',fontsize=9,color='#475467')
save(fig,'revision_amortisation')

fig,axes=plt.subplots(1,2,figsize=(7.4,4),layout='constrained')
for ax,family,title in zip(axes,['zz','exchange'],['(a) Dephasing','(b) Exchange']):
    df=selected(family);base=df[df.role=='dfe']
    for role in ['constant','ridge','neural']:
        d=df[df.role==role].merge(base,on='target',suffixes=('','_base'))
        d=d[d.selected&d.selected_base].sort_values('target');y=100*(1-d.shots/d.shots_base)
        ax.plot(d.target,y,color=C[role],lw=1.7,label=role.title())
        for (_,r),v in zip(d.iterrows(),y):
            passed=r.mean_risk_target_pass and r.mean_risk_target_pass_base
            ax.plot(r.target,v,'o',ms=4.5,color=C[role],mfc=C[role] if passed else 'white')
    ax.set(title=title,xlabel='Target MAE',ylabel='Selected query-shot reduction (%)',xticks=np.arange(.03,.081,.01),ylim=(-3,102))
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
save(fig,'revision_target_sweep')

fig,axes=plt.subplots(1,2,figsize=(7.4,4),layout='constrained')
df=pd.read_csv(ROOT/'two_bath/validation/summary.csv')
for role in ['dfe','constant','ridge']:
    d=df[df.method==METHOD[role]].sort_values('shots')
    axes[0].plot(d.shots,d.mae,color=C[role],label='DFE' if role=='dfe' else role.title()+' shrinkage')
axes[0].axhline(df[df.method=='summary_ridge_prior'].mae.iloc[0],color=C['ridge'],ls=':',label='Ridge prior: 0 shots')
axes[0].set(xlabel='Query shots',ylabel='Validation MAE',title='(a) Two-qubit bath')
audit=pd.read_csv('results_revision/stage4/two_bath/audit.csv')
axes[1].hist(audit.span113,bins=24,color='#6876B9',alpha=.8,edgecolor='white')
axes[1].set(xlabel='Fixed-input span',ylabel='Number of sequences',title='(b) Memory ambiguity')
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
save(fig,'revision_two_bath')

fig,axes=plt.subplots(1,2,figsize=(7.4,4),layout='constrained')
for ax,family,title in zip(axes,['exchange','two_bath'],['(a) Exchange (one bath qubit)','(b) Two-qubit bath']):
    df=pd.read_csv(ROOT/family/'validation/summary.csv')
    for role in ['constant','ridge']:
        for suffix,style,label in [('_eb','--','Legacy'),('_iv','-','Independent')]:
            method=METHOD[role][:-3]+suffix
            d=df[(df.method==method)&(df.allocation=='fixed')&(df.shots<=24)].sort_values('shots')
            ax.plot(d.shots,d.mae,ls=style,color=C[role],marker='o',ms=3,label=role.title()+': '+label)
    ax.set(xlabel='Query shots',ylabel='Validation MAE',title=title,xticks=[4,8,12,16,20,24])
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
save(fig,'revision_weight_diagnostics')

rows=[];deployment=pd.read_csv(ROOT/'zz/deployment_variability.csv')
for t in [.065,.053]:
    for role in ['dfe','constant','ridge','neural']:
        s=head('zz',t,role);r=deployment[(deployment.target==t)&(deployment.method==s.method)&
            (deployment.allocation==s.allocation)&(deployment.shots==s.shots)].iloc[0]
        rows.append([f'{t:.3f}',role.upper() if role=='dfe' else role.title(),str(int(s.shots)),f'{r.mae:.5f}',f'{r.p95:.5f}',f'{100*r.pass_fraction:.1f}\\%'])
table('deployment_table.tex','llrrrr',['Target','Prior','Shots','MAE','95th percentile','Pass fraction'],rows)
for family,name in [('exchange','exchange_followup_table.tex'),('two_bath','two_bath_followup_table.tex')]:
    rows=[]
    for t in [.065,.053]:
        for role in ['dfe','constant','ridge']+(['neural'] if family=='exchange' else []):
            r=head(family,t,role)
            rows.append([f'{t:.3f}',role.upper() if role=='dfe' else role.title(),str(int(r.shots)),f'{r.mae:.5f}',f'{r.upper95:.6f}'])
    table(name,'llrrr',['Target','Method','Shots','Test MAE','Upper 95\\%'],rows)
rows=[]
for family,label in [('zz','Dephasing'),('exchange','Exchange'),('two_bath','Two baths')]:
    df=pd.read_csv(ROOT/family/'weight_diagnostics.csv')
    for _,r in df.iterrows():
        role='Neural' if r.method.startswith('neural') else 'Ridge' if r.method.startswith('summary') else 'Constant'
        wanted=(family=='zz' and role in ['Neural','Ridge']) or (family!='zz' and role in ['Ridge','Constant'] and r.shots in [4,8])
        if wanted:rows.append([label,role,str(int(r.shots)),f'{r.legacy_mae:.5f}',f'{r.new_mae:.5f}',f'{r.difference:+.5f}'])
table('independent_weight_diagnostics_table.tex','llrrrr',['Family','Prior','Shots','Legacy','Independent','Change'],rows)
print('Saved five figures and four tables from stage-five results')
