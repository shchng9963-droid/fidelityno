"""Stage-five paired inference with a fixed comparison family."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.analyze_revision_confirmation import tensor

def weights(nq,seed,nboot=4000):
    rng=np.random.default_rng(seed);w=np.zeros((nboot,20,5),dtype=np.float32)
    for i in range(nboot):
        ds=rng.integers(20,size=20);rs=rng.integers(5,size=(20,5))
        np.add.at(w[i],(np.repeat(ds,5),rs.ravel()),1/100)
    sw=rng.multinomial(5,np.full(5,.2),size=nboot).astype(np.float32)/5
    qw=rng.multinomial(nq,np.full(nq,1/nq),size=nboot).astype(np.float32)/nq
    return w.reshape(nboot,100),np.einsum('bdr,bs->bdrs',w,sw).reshape(nboot,500),qw

def main(family):
    design=json.loads(Path('revision/stage5_design.json').read_text());root=Path('results_revision/stage5')/family
    rows=pd.read_csv(root/'test/runs.csv');arc=np.load(root/'test/paired_errors.npz');err=arc['absolute_errors']
    rowmap={int(r):i for i,r in enumerate(arc['row_ids'])};lock=json.loads((root/'selection.json').read_text())
    w,ws,qw=weights(err.shape[1],design['bootstrap_seed']);boots={};records=[];deployment=[]
    for (m,a,b),sub in rows.groupby(['method','allocation','shots']):
        x=tensor(rows,err,rowmap,m,a,b);flat=x.reshape(-1,x.shape[-1]);ww=ws if x.shape[2]==5 else w
        boot=np.concatenate([((ww[s:s+64]@flat)*qw[s:s+64]).sum(1) for s in range(0,len(ww),64)])
        key=(m,a,int(b));boots[key]=boot
        rec=dict(method=m,allocation=a,shots=int(b),mae=float(x.mean()),low=float(np.quantile(boot,.025)),
            high=float(np.quantile(boot,.975)),upper95=float(np.quantile(boot,.95)))
        records.append(rec);single=x.mean(-1).ravel();groupseed=x.mean(axis=(1,3)).ravel()
        for t in design['headline_targets']:
            deployment.append(dict(**rec,target=t,single_fit_n=len(single),median=float(np.median(single)),
                p05=float(np.quantile(single,.05)),p95=float(np.quantile(single,.95)),pass_fraction=float((single<=t).mean()),
                group_checkpoint_n=len(groupseed),group_checkpoint_p95=float(np.quantile(groupseed,.95)),
                group_checkpoint_pass_fraction=float((groupseed<=t).mean())))
        print(family,m,a,b,rec['mae'],flush=True)
    results=pd.DataFrame(records);results.to_csv(root/'risk_intervals.csv',index=False)
    pd.DataFrame(deployment).to_csv(root/'deployment_variability.csv',index=False)
    selected=[]
    for choice in lock['selections']:
        c=choice['configuration']
        if c is None:selected.append(dict(target=choice['target'],role=choice['role'],selected=False));continue
        r=results[(results.method==c['method'])&(results.allocation==c['allocation'])&(results.shots==c['shots'])].iloc[0].to_dict()
        selected.append(dict(target=choice['target'],role=choice['role'],selected=True,offline_shots=c['offline_shots'],
            mean_risk_target_pass=r['upper95']<=choice['target'],**r))
    curve=pd.DataFrame(selected);curve.to_csv(root/'selected_results.csv',index=False)
    contrasts=[]
    for t in design['headline_targets']:
        for role in (['ridge'] if family=='two_bath' else ['neural','ridge']):
            for base in ['dfe','constant','ridge']:
                if role==base:continue
                pair=curve[(curve.target==t)&curve.role.isin([role,base])]
                if len(pair)!=2 or not pair.selected.all():continue
                h=pair[pair.role==role].iloc[0];b=pair[pair.role==base].iloc[0]
                delta=boots[(h.method,h.allocation,int(h.shots))]-boots[(b.method,b.allocation,int(b.shots))]
                primary=family=='two_bath' or role=='neural';divisor=(4 if family=='two_bath' else 6) if primary else 1
                upper=float(np.quantile(delta,1-.05/divisor));saving=int(b.shots-h.shots)
                common=bool(h.mean_risk_target_pass and b.mean_risk_target_pass)
                valid=common and upper<.002 and saving>0
                extra=int(h.offline_shots-b.offline_shots)
                contrasts.append(dict(target=t,method=role,baseline=base,query_saving=saving,difference=float(h.mae-b.mae),
                    low=float(np.quantile(delta,.025)),high=float(np.quantile(delta,.975)),
                    primary_family=primary,bonferroni_divisor=divisor,upper_fwer=upper,
                    common_mean_target=common,noninferior_001=upper<.001,noninferior_002=upper<.002,
                    noninferior_003=upper<.003,validated_shot_saving=valid,
                    first_strict_saving_query=max(1,int(np.floor(extra/saving))+1) if valid else None))
    pd.DataFrame(contrasts).to_csv(root/'paired_contrasts.csv',index=False)
    diagnostic=[]
    for (m,a,b),new in boots.items():
        if not m.endswith('_iv'):continue
        oldkey=(m[:-3]+'_eb',a,b)
        if oldkey not in boots:continue
        delta=new-boots[oldkey]
        n=results[(results.method==m)&(results.allocation==a)&(results.shots==b)].iloc[0]
        o=results[(results.method==oldkey[0])&(results.allocation==a)&(results.shots==b)].iloc[0]
        diagnostic.append(dict(method=m,allocation=a,shots=b,new_mae=n.mae,legacy_mae=o.mae,
            difference=n.mae-o.mae,low=float(np.quantile(delta,.025)),high=float(np.quantile(delta,.975))))
    pd.DataFrame(diagnostic).to_csv(root/'weight_diagnostics.csv',index=False)
    np.savez_compressed(root/'bootstrap_means.npz',**{'__'.join(map(str,k)):v for k,v in boots.items()})
    print(curve[curve.target.isin(design['headline_targets'])].to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--family',required=True);a=ap.parse_args();main(a.family)
