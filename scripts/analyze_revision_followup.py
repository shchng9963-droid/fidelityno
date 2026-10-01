"""Shared bootstrap of mean risk, and separate single-fit risk distributions."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.analyze_revision_confirmation import tensor

def weights(nq):
    rng=np.random.default_rng(202610031);n=2000
    w=np.zeros((n,20,5),dtype=np.float32)
    for i in range(n):
        ds=rng.integers(20,size=20);rs=rng.integers(5,size=(20,5))
        np.add.at(w[i],(np.repeat(ds,5),rs.ravel()),1/100)
    sw=rng.multinomial(5,np.full(5,.2),size=n).astype(np.float32)/5
    qw=rng.multinomial(nq,np.full(nq,1/nq),size=n).astype(np.float32)/nq
    return w.reshape(n,100),np.einsum('bdr,bs->bdrs',w,sw).reshape(n,500),qw

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',required=True);a=ap.parse_args()
    root=Path('results_revision/stage4')/a.family
    rows=pd.read_csv(root/'runs.csv');arc=np.load(root/'paired_errors.npz');err=arc['absolute_errors']
    rowmap={int(r):i for i,r in enumerate(arc['row_ids'])};lock=json.loads((root/'selection.json').read_text())
    w,ws,qw=weights(err.shape[1]);boots={};records=[];deployment=[]
    for (m,alloc,b),sub in rows[rows.cal_draw>=0].groupby(['method','allocation','shots']):
        x=tensor(rows,err,rowmap,m,alloc,b);flat=x.reshape(-1,x.shape[-1]);ww=ws if x.shape[2]==5 else w
        boot=np.concatenate([((ww[s:s+64]@flat)*qw[s:s+64]).sum(1) for s in range(0,2000,64)])
        key=(m,alloc,int(b));boots[key]=boot
        record=dict(method=m,allocation=alloc,shots=b,mae=float(x.mean()),low=float(np.quantile(boot,.025)),
            high=float(np.quantile(boot,.975)),upper95=float(np.quantile(boot,.95)))
        records.append(record)
        single=x.mean(-1).ravel();groupseed=x.mean(axis=(1,3)).ravel()
        for target in [.065,.053]:
            deployment.append(dict(**record,target=target,single_fit_n=len(single),p05=float(np.quantile(single,.05)),
                median=float(np.median(single)),p95=float(np.quantile(single,.95)),pass_fraction=float((single<=target).mean()),
                group_checkpoint_n=len(groupseed),group_checkpoint_p95=float(np.quantile(groupseed,.95)),
                group_checkpoint_pass_fraction=float((groupseed<=target).mean())))
        print(a.family,m,alloc,b,record['mae'],flush=True)
    results=pd.DataFrame(records);results.to_csv(root/'risk_intervals.csv',index=False)
    pd.DataFrame(deployment).to_csv(root/'deployment_variability.csv',index=False)
    curves=[];contrasts=[]
    for s in lock['selections']:
        c=s['configuration']
        if c is None:curves.append(dict(target=s['target'],role=s['role'],selected=False));continue
        r=results[(results.method==c['method'])&(results.allocation==c['allocation'])&(results.shots==c['shots'])].iloc[0].to_dict()
        curves.append(dict(target=s['target'],role=s['role'],selected=True,offline_shots=c['offline_shots'],
            mean_risk_target_pass=r['upper95']<=s['target'],**r))
    curve=pd.DataFrame(curves);curve.to_csv(root/'selected_results.csv',index=False)
    roles=['neural','ridge'] if a.family!='two_bath' else ['ridge']
    for target in [.065,.053]:
        for role in roles:
            for base in ['constant','ridge','dfe']:
                if role==base:continue
                pair=curve[(curve.target==target)&curve.role.isin([role,base])]
                if len(pair)!=2 or not pair.selected.all():continue
                h=pair[pair.role==role].iloc[0];b=pair[pair.role==base].iloc[0]
                delta=boots[(h.method,h.allocation,int(h.shots))]-boots[(b.method,b.allocation,int(b.shots))]
                contrasts.append(dict(target=target,method=role,baseline=base,query_saving=int(b.shots-h.shots),
                    difference=float(h.mae-b.mae),low=float(np.quantile(delta,.025)),high=float(np.quantile(delta,.975)),
                    common_mean_target=bool(h.mean_risk_target_pass and b.mean_risk_target_pass),
                    noninferior_001=bool(np.quantile(delta,.975)<.001),noninferior_002=bool(np.quantile(delta,.975)<.002),
                    noninferior_003=bool(np.quantile(delta,.975)<.003)))
    pd.DataFrame(contrasts).to_csv(root/'paired_contrasts.csv',index=False)
    np.savez_compressed(root/'bootstrap_means.npz',**{'__'.join(map(str,k)):v for k,v in boots.items()})
    print(curve[curve.target.isin([.065,.053])].to_string(index=False),flush=True)
if __name__=='__main__':main()
