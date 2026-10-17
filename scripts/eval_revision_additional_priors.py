"""Post-hoc simple-prior controls on the final independent-weight protocol.

Run simulations on the experimental server. Original data/results are read-only.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.eval_revision_confirmatory import fit,digest
from scripts.analyze_revision_independent_weights import weights
from physics.baselines.revision_measurement import (sample_measurement,
    calibration_outcome_variances,independent_measurement_variance,
    residual_prior_variance,empirical_bayes_fusion)

DESIGN=Path('revision/additional_priors_design_20261017.json')
ROOT=Path('results_revision/additional_priors_20261017')
NAMES=['constant_median','product_affine','exact_marginal_affine','constant_mean','summary_ridge']

def load(family,split):
    p=(Path('results_revision/stage5')/family/'cache_test.npz' if split=='test' else
       Path('results_revision/stage2')/family/('cache_'+split+'.npz'))
    with np.load(p,allow_pickle=False) as z:return dict(z),p

def evaluate(family,split):
    root=ROOT/family;dest=root/split
    if dest.exists():raise FileExistsError(dest)
    design=json.loads(DESIGN.read_text());stage=json.loads(Path('revision/stage5_design.json').read_text())
    cal,cp=load(family,'calibration');query,qp=load(family,split)
    assert not set(cal['x_fingerprints'])&set(query['x_fingerprints'])
    assert len(cal['y'])==1280 and len(query['y'])==4096
    lock=None;needed=None
    if split=='test':
        lock=json.loads((root/'selection.json').read_text())
        assert lock['design_sha256']==digest(DESIGN)
        assert lock['script_sha256']==digest(__file__)
        assert lock['validation_sha256']==digest(root/'validation/summary.csv')
        needed={(c['method'],c['allocation'],c['shots']) for r in lock['selections']
                for c in [r['positive'],r['including_zero']] if c}
        # Original locked control configurations are included solely for exact
        # regression checks, irrespective of our zero-inclusive selections.
        original=json.loads((Path('results_revision/stage5')/family/'selection.json').read_text())
        for r in original['selections']:
            if r['target'] in design['targets'] and r['role'] in ['constant','ridge']:
                c=r['configuration'];needed.add((c['method'],c['allocation'],c['shots']))
    allocations=['fixed','pilot']+(['zz_known'] if family=='zz' else [])
    rows=[];sums={};errors=[];ids=[];moments=[];n=len(query['y']);allq=np.arange(n)
    def add(name,a,b,g,r,pred):
        e=np.abs(np.clip(pred,0,1)-query['y']);key=(name,a,b)
        rows.append(dict(method=name,allocation=a,shots=b,cal_draw=g,repeat=r,
                         checkpoint='none',mae=float(e.mean()),offline_shots=4096))
        if key not in sums:sums[key]=[np.zeros(n),0]
        sums[key][0]+=e;sums[key][1]+=1
        if split=='test':errors.append(e.astype(np.float32));ids.append(len(rows)-1)
    seed={'zz':202610101,'exchange':202610102}[family]
    for g in range(20):
        idx=np.arange(g*64,(g+1)*64)
        for r in range(5):
            label,lv,counts,detail=sample_measurement(cal['ex'][idx],64,
                np.random.default_rng(np.random.SeedSequence([seed,g,r,1])),
                'zz_known' if family=='zz' else 'fixed',return_observations=True)
            assert np.all(counts==64)
            vcal=calibration_outcome_variances(detail)
            moments.append(dict(cal_draw=g,repeat=r,label_min=float(label.min()),
                label_max=float(label.max()),negative_labels=int((label<0).sum()),
                above_one_labels=int((label>1).sum()),variance=vcal.tolist()))
            priors={}
            for name in NAMES:
                oof=np.zeros(64)
                for fold in range(4):
                    held=np.arange(64)%4==fold
                    oof[held]=fit(name,cal,cal,idx[~held],label[~held],idx[held])
                pred=fit(name,cal,query,idx,label,allq)
                priors[name]=(pred,residual_prior_variance(oof,label,lv))
                add(name+'_prior','none',0,g,r,pred)
            for ai,a in enumerate(allocations):
                for b in stage['budgets']:
                    wanted=[name for name in NAMES if needed is None or (name+'_iv',a,b) in needed]
                    if not wanted:continue
                    stream=([seed,g,r,ai,b,3] if split=='validation' else
                            [stage['test_measurement_seeds'][family],g,r,ai,b,4])
                    obs,_,counts,detail=sample_measurement(query['ex'],b,
                        np.random.default_rng(np.random.SeedSequence(stream)),a,return_observations=True)
                    assert np.all(counts==b)
                    settings=vcal if a=='zz_known' or family!='zz' else np.r_[vcal,0.]
                    v=independent_measurement_variance(settings,detail['setting_shots'])
                    for name in wanted:
                        prior,tau=priors[name]
                        add(name+'_iv',a,b,g,r,empirical_bayes_fusion(prior,obs,v,tau))
        print(family,split,'group',g+1,flush=True)
    runs=pd.DataFrame(rows);summary=[]
    for (m,a,b),(summed,count) in sums.items():
        sub=runs[(runs.method==m)&(runs.allocation==a)&(runs.shots==b)]
        q=summed/count;groups=sub.groupby('cal_draw').mae.mean().to_numpy()
        qterm=q.var(ddof=1)/n;gterm=groups.var(ddof=1)/20
        summary.append(dict(method=m,allocation=a,shots=b,mae=float(q.mean()),
            query_component=float(qterm),group_component=float(gterm),checkpoint_component=0.,
            selection_score=float(q.mean()+1.645*np.sqrt(qterm+gterm)),offline_shots=4096))
    df=pd.DataFrame(summary)
    priorfile=Path('results_revision/stage5')/family/split/'summary.csv'
    old=pd.read_csv(priorfile);checks=[]
    for row in df[df.method.str.startswith(('constant_mean','summary_ridge'))].itertuples():
        matched=old[(old.method==row.method)&(old.allocation==row.allocation)&(old.shots==row.shots)]
        if not len(matched):continue
        d=abs(row.mae-float(matched.iloc[0].mae));assert d<1e-12,(family,split,row.method,d)
        if split=='validation':assert abs(row.selection_score-float(matched.iloc[0].selection_score))<1e-12
        checks.append(dict(method=row.method,allocation=row.allocation,shots=row.shots,absolute_difference=d))
    dest.mkdir(parents=True);runs.to_csv(dest/'runs.csv',index=False);df.to_csv(dest/'summary.csv',index=False)
    (dest/'calibration_moments.json').write_text(json.dumps(moments,indent=2))
    (dest/'regression_checks.json').write_text(json.dumps(checks,indent=2))
    if split=='test':np.savez_compressed(dest/'paired_errors.npz',absolute_errors=np.asarray(errors),row_ids=ids)
    (dest/'manifest.json').write_text(json.dumps(dict(status=design['status'],
        design_sha256=digest(DESIGN),script_sha256=digest(__file__),calibration_cache_sha256=digest(cp),
        query_cache_sha256=digest(qp),core_sha256={p:digest(p) for p in [
            'physics/baselines/revision_measurement.py','scripts/eval_revision_confirmatory.py']},
        original_checkpoint_provenance=json.loads(str(cal['provenance']))),indent=2))
    if split=='validation':
        def choose(sub):
            if not len(sub):return None
            row=sub.sort_values(['shots','offline_shots','selection_score','allocation']).iloc[0]
            return {k:int(row[k]) if k in ['shots','offline_shots'] else float(row[k]) if k in ['mae','selection_score'] else row[k]
                    for k in ['method','allocation','shots','offline_shots','mae','selection_score']}
        selected=[]
        for t in design['targets']:
            for name in NAMES:
                eligible=df[(df.method==name+'_iv')&(df.selection_score<=t)]
                zero=df[(df.method==name+'_prior')&(df.selection_score<=t)]
                selected.append(dict(target=t,prior=name,positive=choose(eligible),
                                     including_zero=choose(pd.concat([eligible,zero]))))
        (root/'selection.json').write_text(json.dumps(dict(status=design['status'],
            design_sha256=digest(DESIGN),script_sha256=digest(__file__),
            validation_sha256=digest(dest/'summary.csv'),selections=selected),indent=2))

def analyze(family):
    root=ROOT/family;rows=pd.read_csv(root/'test/runs.csv');lock=json.loads((root/'selection.json').read_text())
    with np.load(root/'test/paired_errors.npz') as z:err=z['absolute_errors'];ids=z['row_ids']
    rowmap={int(r):i for i,r in enumerate(ids)};w,_,qw=weights(err.shape[1],202610221)
    estimates=[]
    for (m,a,b),sub in rows.groupby(['method','allocation','shots']):
        sub=sub.sort_values(['cal_draw','repeat']);x=err[[rowmap[int(i)] for i in sub.index]]
        assert x.shape==(100,4096)
        boot=np.concatenate([((w[i:i+64]@x)*qw[i:i+64]).sum(1) for i in range(0,4000,64)])
        estimates.append(dict(method=m,allocation=a,shots=int(b),test_mae=float(sub.mae.mean()),
            low95=float(np.quantile(boot,.025)),high95=float(np.quantile(boot,.975)),upper95=float(np.quantile(boot,.95))))
    risks=pd.DataFrame(estimates);risks.to_csv(root/'risk_intervals.csv',index=False);out=[]
    for r in lock['selections']:
        for rule in ['positive','including_zero']:
            c=r[rule]
            if c is None:out.append(dict(target=r['target'],prior=r['prior'],selection=rule,selected=False));continue
            row=risks[(risks.method==c['method'])&(risks.allocation==c['allocation'])&(risks.shots==c['shots'])].iloc[0].to_dict()
            out.append(dict(family=family,target=r['target'],prior=r['prior'],selection=rule,selected=True,
                calibration_shots=4096,fusion_rule='independent-weight shrinkage' if c['shots'] else 'zero-query prior',
                validation_mae=c['mae'],guard=c['selection_score'],target_pass=row['upper95']<=r['target'],**row))
    pd.DataFrame(out).to_csv(root/'selected_results.csv',index=False)
    print(pd.DataFrame(out).to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--family',choices=['zz','exchange'],required=True)
    ap.add_argument('--mode',choices=['validation','test','analyze'],required=True);a=ap.parse_args()
    analyze(a.family) if a.mode=='analyze' else evaluate(a.family,a.mode)
