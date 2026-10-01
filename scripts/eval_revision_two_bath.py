"""Validation-locked non-neural extension to a four-dimensional bath."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from physics.channels.two_bath_memory import BATH,LOW,HIGH,unitary,sample_parameters,marginal_superop,fidelity_from_superop
from physics.channels.batched_memory import memory_grid_choi,memory_grid_fidelity
from physics.baselines.hybrid import batch_pauli_expectations_from_choi
from physics.baselines.revision_measurement import sample_measurement,residual_prior_variance,empirical_bayes_fusion
from scripts.eval_revision_confirmatory import fit,digest
ROOT=Path('results_revision/stage4/two_bath')
DATA=Path('data_revision/stage4/two_bath')

def generate(split):
    design=json.loads(Path('revision/stage4_design.json').read_text())['larger_bath']
    DATA.mkdir(parents=True,exist_ok=True);ROOT.mkdir(parents=True,exist_ok=True)
    dest=DATA/(split+'.npz')
    if dest.exists(): raise FileExistsError(dest)
    if split=='test' and not (ROOT/'selection.json').exists():raise RuntimeError('lock before test generation')
    si=['calibration','validation','test'].index(split)
    rng=np.random.default_rng(design[split+'_parameter_seed']);erng=np.random.default_rng(design['eta_seeds'][si])
    n=design[split+'_n'];length=rng.choice([8,16,24],size=n);eta=erng.uniform(.85,.99,n)
    ys=[];ex=[];features=[];params=[];fingerprints=[];checks=[]
    for i,l in enumerate(length):
        p=sample_parameters(rng,l);us=[unitary(t) for t in p];s=np.eye(4,dtype=complex);pf=[]
        for u in us:
            step,j=marginal_superop(u);s=step@s;pf.append(fidelity_from_superop(step))
        exact=fidelity_from_superop(s);prod=np.prod(pf);inf=1-np.array(pf)
        features.append([l,prod,exact,inf.sum(),inf.mean(),inf.std(),inf.min(),inf.max()])
        j=memory_grid_choi(us,[eta[i],0],BATH)
        checks.append(abs(memory_grid_fidelity(us,[0],BATH)[0]-exact))
        r=batch_pauli_expectations_from_choi(j[:1])[0];ex.append(r);ys.append(r.mean())
        padded=np.full((24,8),np.nan);padded[:l]=p;params.append(padded)
        fingerprints.append(hashlib.sha256(p.tobytes()).hexdigest())
        if (i+1)%512==0:print('two bath',split,i+1,flush=True)
    assert max(checks)<1e-10
    np.savez_compressed(dest,y=ys,ex=ex,features=features,length=length,eta=eta,parameters=params,
        x_fingerprints=fingerprints,product_affine=np.array(features)[:,1],exact_marginal_affine=np.array(features)[:,2])
    (ROOT/(split+'_data_manifest.json')).write_text(json.dumps(dict(design_sha256=digest('revision/stage4_design.json'),
        physical_source_sha256=digest('physics/channels/two_bath_memory.py'),parameter_low=LOW.tolist(),parameter_high=HIGH.tolist(),
        max_reset_composition_error=max(checks),data_sha256=digest(dest)),indent=2))

def audit():
    rng=np.random.default_rng(202610054);grid=np.linspace(.85,.99,113);rows=[];pars=[];values=[]
    for i in range(512):
        l=int(rng.choice([8,16,24]));p=sample_parameters(rng,l);v=memory_grid_fidelity([unitary(t) for t in p],grid,BATH)
        rows.append(dict(base=i,length=l,span15=np.ptp(v[::8]),span113=np.ptp(v),discrete_risk113=np.abs(v-np.median(v)).mean()))
        pad=np.full((24,8),np.nan);pad[:l]=p;pars.append(pad);values.append(v)
        if (i+1)%128==0:print('two bath audit',i+1,flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'audit.csv',index=False)
    np.savez_compressed(DATA/'audit.npz',parameters=pars,eta=grid,fidelity=values)
    print(pd.DataFrame(rows).mean().to_dict(),flush=True)

def evaluate(split):
    design=json.loads(Path('revision/stage4_design.json').read_text());larger=design['larger_bath']
    root=ROOT/split;root.mkdir(parents=True,exist_ok=True)
    if (root/'summary.csv').exists():raise FileExistsError(root)
    cal=dict(np.load(DATA/'calibration.npz'));test=dict(np.load(DATA/(split+'.npz')))
    assert not(set(cal['x_fingerprints'])&set(test['x_fingerprints']))
    if split=='test':
        val=np.load(DATA/'validation.npz');assert not(set(val['x_fingerprints'])&set(test['x_fingerprints']))
        lock=json.loads((ROOT/'selection.json').read_text())
        assert lock['validation_sha256']==digest(ROOT/'validation/summary.csv')
        configs={(r['configuration']['method'],r['configuration']['shots']) for r in lock['selections'] if r['configuration']}
    else:configs={(m,b) for m in larger['methods'] for b in larger['budgets']}
    rows=[];errors=[];n=len(test['y']);allq=np.arange(n)
    def add(method,b,d,r,pred):
        e=np.abs(np.clip(pred,0,1)-test['y']).astype(np.float32);errors.append(e)
        rows.append(dict(method=method,shots=b,allocation='fixed' if b else 'none',checkpoint='none',
            cal_draw=d,repeat=r,mae=float(e.mean()),offline_shots=0 if method=='dfe' else 4096))
    for d in range(20):
        ix=np.arange(d*64,(d+1)*64)
        for r in range(5):
            lab,lv,_=sample_measurement(cal['ex'][ix],64,np.random.default_rng(np.random.SeedSequence([202610055,d,r,1])),'fixed')
            priors={}
            for name in ['constant_mean','summary_ridge','product_affine','exact_marginal_affine']:
                oof=np.zeros(64)
                for fold in range(4):
                    held=np.arange(64)%4==fold;oof[held]=fit(name,cal,cal,ix[~held],lab[~held],ix[held])
                prior=fit(name,cal,test,ix,lab,allq);priors[name]=(prior,residual_prior_variance(oof,lab,lv))
                add(name+'_prior',0,d,r,prior)
            for b in sorted({b for _,b in configs}):
                obs,var,_=sample_measurement(test['ex'],b,np.random.default_rng(np.random.SeedSequence([202610055,d,r,b,2 if split=='validation' else 3])),'fixed')
                if ('dfe',b) in configs:add('dfe',b,d,r,obs)
                for name,(p,tau) in priors.items():
                    if (name+'_eb',b) in configs:add(name+'_eb',b,d,r,empirical_bayes_fusion(p,obs,var,tau))
        print('two bath',split,'group',d+1,flush=True)
    df=pd.DataFrame(rows);df.to_csv(root/'runs.csv',index=False);errors=np.asarray(errors)
    summaries=[]
    for (m,a,b),sub in df.groupby(['method','allocation','shots']):
        e=errors[sub.index].mean(0);dm=sub.groupby('cal_draw').mae.mean().values
        guard=e.mean()+1.645*np.sqrt(e.var(ddof=1)/n+dm.var(ddof=1)/len(dm))
        summaries.append(dict(method=m,allocation=a,shots=b,mae=float(e.mean()),selection_score=float(guard),offline_shots=int(sub.offline_shots.max())))
    summary=pd.DataFrame(summaries);summary.to_csv(root/'summary.csv',index=False)
    if split=='validation':
        chosen=[]
        for target in design['targets']:
            for role,m in [('dfe','dfe'),('constant','constant_mean_eb'),('ridge','summary_ridge_eb')]:
                ok=summary[(summary.method==m)&(summary.selection_score<=target)].sort_values(['shots','selection_score'])
                c=None if ok.empty else json.loads(ok.iloc[0].to_json())
                chosen.append(dict(target=target,role=role,configuration=c))
        (ROOT/'selection.json').write_text(json.dumps(dict(design_sha256=digest('revision/stage4_design.json'),validation_sha256=digest(root/'summary.csv'),selections=chosen),indent=2))
    else:
        np.savez_compressed(ROOT/'paired_errors.npz',absolute_errors=errors,row_ids=np.arange(len(df)),truth=test['y'],lengths=test['length'])
        df.to_csv(ROOT/'runs.csv',index=False);summary.to_csv(ROOT/'summary.csv',index=False)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['development','test'],required=True);a=ap.parse_args()
    if a.mode=='development':generate('calibration');generate('validation');audit();evaluate('validation')
    else:generate('test');evaluate('test')
