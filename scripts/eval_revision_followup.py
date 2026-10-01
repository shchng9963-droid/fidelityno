"""Per-class validation locks and a fresh test after external review."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.eval_revision_confirmatory import digest,prepare,fit,canonical
from physics.baselines.revision_measurement import sample_measurement,residual_prior_variance,empirical_bayes_fusion


def select(family):
    design=json.loads(Path('revision/stage4_design.json').read_text())
    root=Path('results_revision/stage4')/family; root.mkdir(parents=True,exist_ok=True)
    if (root/'selection.json').exists(): raise FileExistsError('selection already locked')
    if Path('data_revision/stage4',family,'test.npz').exists(): raise RuntimeError('lock before fresh test generation')
    val=Path('results_revision/stage2')/family/'validation/summary.csv'
    df=pd.read_csv(val); selected=[]
    for target in design['targets']:
        for role,method in design['classes'].items():
            eligible=df[(df.method==method)&(df.selection_score<=target)]
            item=dict(target=target,role=role,configuration=None)
            if not eligible.empty:
                r=eligible.sort_values(['shots','offline_shots','selection_score','allocation']).iloc[0]
                item['configuration']=dict(method=method,allocation=r.allocation,shots=int(r.shots),
                    offline_shots=int(r.offline_shots),validation_mae=float(r.mae),guard=float(r.selection_score))
            selected.append(item)
    payload=dict(design_sha256=digest('revision/stage4_design.json'),validation_sha256=digest(val),selections=selected)
    (root/'selection.json').write_text(json.dumps(payload,indent=2))
    print(family,[(r['target'],r['role'],None if r['configuration'] is None else r['configuration']['shots']) for r in selected],flush=True)


def evaluate(family,ckpts):
    root=Path('results_revision/stage4')/family
    if (root/'summary.csv').exists(): raise FileExistsError('test complete')
    design=json.loads(Path('revision/stage4_design.json').read_text())
    lock=json.loads((root/'selection.json').read_text())
    if lock['design_sha256']!=digest('revision/stage4_design.json'): raise RuntimeError('design changed')
    if lock['validation_sha256']!=digest(Path('results_revision/stage2')/family/'validation/summary.csv'): raise RuntimeError('validation changed')
    configurations={(c['method'],c['allocation'],c['shots']) for r in lock['selections'] if (c:=r['configuration'])}
    oldroot=Path('results_revision/stage2')/family
    cal=prepare(Path('data_revision')/family/'calibration.npz',ckpts,oldroot/'cache_calibration.npz')
    data=Path('data_revision/stage4')/family/'test.npz'
    test=prepare(data,ckpts,root/'cache_test.npz')
    for previous in [cal,*[dict(np.load(oldroot/('cache_'+s+'.npz'))) for s in ['validation','test']]]:
        assert not(set(previous['x_fingerprints']) & set(test['x_fingerprints']))
    names=['constant_mean','constant_median','summary_ridge','product_affine','exact_marginal_affine']+['neural_'+Path(p).stem for p in ckpts]
    rows=[]; errors=[]; n=len(test['y']); allq=np.arange(n)
    seed=202610101 if family=='zz' else 202610102
    def add(name,alloc,b,draw,rep,pred):
        error=np.abs(np.clip(pred,0,1)-test['y']).astype(np.float32)
        rows.append(dict(method=canonical(name),checkpoint=name if name.startswith('neural_') else 'none',
            allocation=alloc,shots=b,cal_draw=draw,repeat=rep,mae=float(error.mean()),offline_shots=0 if name=='dfe' or name.endswith('_raw') else 4096))
        errors.append(error)
    for name in names[3:]: add(name+'_raw','none',0,-1,-1,test[name])
    allocs=['fixed','pilot']+(['zz_known'] if family=='zz' else [])
    for draw in range(20):
        idx=np.arange(draw*64,(draw+1)*64)
        for rep in range(5):
            labels,lv,_=sample_measurement(cal['ex'][idx],64,np.random.default_rng(np.random.SeedSequence([seed,draw,rep,1])), 'zz_known' if family=='zz' else 'fixed')
            priors={}
            for name in names:
                oof=np.zeros(64)
                for fold in range(4):
                    held=np.arange(64)%4==fold
                    oof[held]=fit(name,cal,cal,idx[~held],labels[~held],idx[held])
                prior=fit(name,cal,test,idx,labels,allq)
                priors[name]=(prior,residual_prior_variance(oof,labels,lv))
                add(name+'_prior','none',0,draw,rep,prior)
            for alloc,b in sorted({(a,b) for _,a,b in configurations}):
                obs,var,counts=sample_measurement(test['ex'],b,np.random.default_rng(np.random.SeedSequence([seed,draw,rep,allocs.index(alloc),b,4002])),alloc)
                if ('dfe',alloc,b) in configurations: add('dfe',alloc,b,draw,rep,obs)
                for name,(prior,tau) in priors.items():
                    if (canonical(name)+'_eb',alloc,b) in configurations:
                        add(name+'_eb',alloc,b,draw,rep,empirical_bayes_fusion(prior,obs,var,tau))
        print(family,'fresh test calibration group',draw+1,flush=True)
    rows=pd.DataFrame(rows); rows.to_csv(root/'runs.csv',index=False)
    rows.groupby(['method','allocation','shots','offline_shots']).mae.mean().reset_index().to_csv(root/'summary.csv',index=False)
    np.savez_compressed(root/'paired_errors.npz',absolute_errors=np.asarray(errors),row_ids=np.arange(len(rows)),lengths=test['length'],truth=test['y'])
    (root/'manifest.json').write_text(json.dumps(dict(selection_sha256=digest(root/'selection.json'),test_sha256=digest(data),
        design_sha256=digest('revision/stage4_design.json'),test_n=n,independent_of_stage2_test=True),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['select','test'],required=True)
    ap.add_argument('--family',choices=['zz','exchange'],required=True); ap.add_argument('--ckpts',nargs='*',default=[])
    args=ap.parse_args()
    select(args.family) if args.mode=='select' else evaluate(args.family,args.ckpts)
