"""Secondary diagnostics after the locked clean-test analysis; no selection.

Run on the experimental server. Readout assumptions are deliberately narrow:
independent symmetric flips with a known rate, including calibration labels.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scripts.eval_revision_confirmatory import fit, canonical, digest
from scripts.eval_recalibrated import predict_quantiles
from physics.baselines.revision_measurement import sample_measurement, residual_prior_variance, empirical_bayes_fusion
from physics.baselines.hybrid import fit_convex_fusion


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family', required=True, choices=['zz', 'exchange'])
    ap.add_argument('--ckpts', nargs='+', required=True)
    a = ap.parse_args()
    root = Path('results_revision/stage3') / a.family
    root.mkdir(parents=True, exist_ok=True)
    if (root/'manifest.json').exists():
        raise FileExistsError('audit already complete')
    source = Path('results_revision/stage2') / a.family
    cal = dict(np.load(source/'cache_calibration.npz', allow_pickle=False))
    test = dict(np.load(source/'cache_test.npz', allow_pickle=False))
    levels = np.arange(1,10)/10
    qrows, covrows, metadata = [], [], []
    for ckpt in a.ckpts:
        q, y = predict_quantiles(ckpt, 'data_revision/'+a.family+'/test.npz', device='cuda')
        residual = y[:,None]-q
        coverage = np.mean(y[:,None] <= q, axis=0)
        pinball = np.maximum(levels*residual, (levels-1)*residual).mean()
        qrows.append(dict(checkpoint=Path(ckpt).stem, n=len(y), mae=float(np.abs(q.mean(1)-y).mean()),
            mean_pinball=float(pinball), quantile_score_9=float(2*pinball),
            quantile_ece=float(np.abs(coverage-levels).mean()),
            central80_coverage=float(np.mean((y>=q[:,0]) & (y<=q[:,-1]))),
            crossing_count=int(np.sum(np.diff(q,axis=1)<0)),
            outside_count=int(np.sum((q<0)|(q>1))),
            clipped_at_one_fraction=float(np.mean(q==1))))
        for level, cov in zip(levels,coverage):
            covrows.append(dict(checkpoint=Path(ckpt).stem, level=level, coverage=cov))
        ck = torch.load(ckpt,map_location='cpu',weights_only=False)
        metadata.append(dict(checkpoint=Path(ckpt).stem, sha256=digest(ckpt),cfg=str(ck['cfg']),
            projection_bias=ck['model']['head.proj.bias'].tolist()))
    pd.DataFrame(qrows).to_csv(root/'raw_uncertainty.csv',index=False)
    pd.DataFrame(covrows).to_csv(root/'raw_coverage.csv',index=False)
    (root/'checkpoints.json').write_text(json.dumps(metadata,indent=2))
    names=['summary_ridge']+['neural_'+Path(c).stem for c in a.ckpts]
    config = [('dfe',32),('dfe',48),('eb',20),('eb',36)] if a.family=='zz' else [('dfe',32),('dfe',92),('fusion',32),('fusion',36)]
    allocation = 'zz_known' if a.family=='zz' else 'fixed'
    rows=[]
    stored=[]
    all_idx=np.arange(len(test['y']))
    seed=202610301 if a.family=='zz' else 202610302
    def add(method,budget,p,draw,rep,pred):
        errors=np.abs(np.clip(pred,0,1)-test['y'])
        rows.append(dict(method=canonical(method),checkpoint=method if method.startswith('neural_') else 'none',
            readout_error=p,shots=budget,cal_draw=draw,repeat=rep,mae=float(errors.mean())))
        stored.append(errors.astype(np.float32))
    for pi,p in enumerate([0.,.01,.03,.05]):
        for draw in range(20):
            idx=np.arange(draw*64,(draw+1)*64)
            for rep in range(5):
                labels,lv,_=sample_measurement(cal['ex'][idx],64,np.random.default_rng(np.random.SeedSequence([seed,pi,draw,rep,0])),allocation,p)
                priors={}
                for name in names:
                    oof=np.zeros(64)
                    for fold in range(4):
                        held=np.arange(64)%4==fold
                        oof[held]=fit(name,cal,cal,idx[~held],labels[~held],idx[held])
                    prior=fit(name,cal,test,idx,labels,all_idx)
                    priors[name]=(prior,oof,residual_prior_variance(oof,labels,lv))
                    add(name+'_prior',0,p,draw,rep,prior)
                for kind,b in config:
                    obs,var,count=sample_measurement(test['ex'],b,np.random.default_rng(np.random.SeedSequence([seed,pi,draw,rep,b,1])),allocation,p)
                    assert np.all(count==b)
                    if kind=='dfe':
                        add('dfe',b,p,draw,rep,obs)
                    else:
                        if kind=='fusion':
                            calobs,_,_=sample_measurement(cal['ex'][idx],b,np.random.default_rng(np.random.SeedSequence([seed,pi,draw,rep,b,2])),allocation,p)
                        for name,(prior,oof,tau) in priors.items():
                            pred=empirical_bayes_fusion(prior,obs,var,tau) if kind=='eb' else prior+fit_convex_fusion(oof,calobs,labels)*(obs-prior)
                            add(name+'_'+kind,b,p,draw,rep,pred)
        print(a.family,'completed flip rate',p,flush=True)
    runs=pd.DataFrame(rows)
    runs.to_csv(root/'readout_runs.csv',index=False)
    runs.groupby(['method','readout_error','shots']).mae.agg(['mean','std','count']).reset_index().to_csv(root/'readout_summary.csv',index=False)
    np.savez_compressed(root/'readout_errors.npz',absolute_errors=np.asarray(stored),row_ids=np.arange(len(rows)))
    (root/'manifest.json').write_text(json.dumps(dict(status='secondary stress test, no post-test budget selection',
        family=a.family,calibration_labels='64 labels of 64 shots, same known readout rate as queries',
        assumptions='independent symmetric flips; known rate; bath reset before each full execution',
        uncertainty='raw nine quantiles; no conformal adjustment or claimed OOD coverage',
        calibration_sha256=digest('data_revision/'+a.family+'/calibration.npz'),
        test_sha256=digest('data_revision/'+a.family+'/test.npz')),indent=2))


if __name__=='__main__':
    main()
