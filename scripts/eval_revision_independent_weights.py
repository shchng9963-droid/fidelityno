"""Stage-five independent-weight validation, immutable locks and fresh tests."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.eval_revision_confirmatory import prepare,fit,canonical,digest
from physics.baselines.revision_measurement import (sample_measurement,residual_prior_variance,
    empirical_bayes_fusion,calibration_outcome_variances,independent_measurement_variance)

DESIGN=Path('revision/stage5_design.json')
ROOT=Path('results_revision/stage5')
DATA=Path('data_revision/stage5')

def code_hashes():
    return {p:digest(p) for p in [str(DESIGN),__file__,
        'physics/baselines/revision_measurement.py','scripts/eval_revision_confirmatory.py']}

def get_data(family,split,ckpts):
    if family=='two_bath':
        path=(DATA/family/'test.npz' if split=='test' else Path('data_revision/stage4/two_bath')/(split+'.npz'))
        with np.load(path) as z:return dict(z)
    old=Path('results_revision/stage2')/family
    path=(DATA/family/'test.npz' if split=='test' else Path('data_revision')/family/(split+'.npz'))
    cache=(ROOT/family/'cache_test.npz' if split=='test' else old/('cache_'+split+'.npz'))
    return prepare(path,ckpts,cache)

def lock(family):
    design=json.loads(DESIGN.read_text());root=ROOT/family
    dest=root/'selection.json';validation=root/'validation/summary.csv'
    if dest.exists() or (DATA/family/'test.npz').exists():raise FileExistsError('already locked or test exists')
    df=pd.read_csv(validation);rows=[]
    classes={'dfe':'dfe','constant':'constant_mean_iv','ridge':'summary_ridge_iv'}
    if family!='two_bath':classes['neural']='neural_iv'
    for target in design['targets']:
        for role,method in classes.items():
            eligible=df[(df.method==method)&(df.selection_score<=target)].sort_values(['shots','offline_shots','selection_score','allocation'])
            config=None
            if len(eligible):
                b=eligible.iloc[0];config=dict(method=method,allocation=b.allocation,shots=int(b.shots),
                    offline_shots=int(b.offline_shots),validation_mae=float(b.mae),guard=float(b.selection_score))
            rows.append(dict(target=target,role=role,configuration=config))
    payload=dict(design_sha256=digest(DESIGN),code_sha256=code_hashes(),validation_sha256=digest(validation),selections=rows)
    dest.write_text(json.dumps(payload,indent=2))
    print(family,'LOCKED',[(r['target'],r['role'],None if r['configuration'] is None else r['configuration']['shots'])
        for r in rows if r['target'] in design['headline_targets']],flush=True)

def verify_lock(family):
    root=ROOT/family;locked=json.loads((root/'selection.json').read_text())
    assert locked['design_sha256']==digest(DESIGN)
    # File identity is the source content, not the absolute checkout location.
    assert sorted(locked['code_sha256'].values())==sorted(code_hashes().values())
    assert locked['validation_sha256']==digest(root/'validation/summary.csv')
    return locked

def generate(family):
    verify_lock(family);d=json.loads(DESIGN.read_text());dest=DATA/family/'test.npz'
    if dest.exists():raise FileExistsError(dest)
    dest.parent.mkdir(parents=True,exist_ok=True);parameter,eta_seed=d['test_seeds'][family]
    if family!='two_bath':
        subprocess.run([sys.executable,'scripts/gen_collision_independent_eta_split.py','--family',family,
            '--out',str(dest),'--n',str(d['test_n']),'--parameter-seed',str(parameter),'--eta-seed',str(eta_seed)],check=True)
        return
    from physics.channels.two_bath_memory import BATH,unitary,sample_parameters,marginal_superop,fidelity_from_superop
    from physics.channels.batched_memory import memory_grid_choi,memory_grid_fidelity
    from physics.baselines.hybrid import batch_pauli_expectations_from_choi
    rng=np.random.default_rng(parameter);erng=np.random.default_rng(eta_seed);n=d['test_n']
    lengths=rng.choice([8,16,24],size=n);eta=erng.uniform(.85,.99,n)
    ys=[];ex=[];features=[];parameters=[];fingerprints=[];checks=[]
    for i,l in enumerate(lengths):
        p=sample_parameters(rng,l);us=[unitary(t) for t in p];s=np.eye(4,dtype=complex);pf=[]
        for u in us:
            step,_=marginal_superop(u);s=step@s;pf.append(fidelity_from_superop(step))
        exact=fidelity_from_superop(s);prod=np.prod(pf);inf=1-np.array(pf)
        features.append([l,prod,exact,inf.sum(),inf.mean(),inf.std(),inf.min(),inf.max()])
        j=memory_grid_choi(us,[eta[i],0],BATH)
        checks.append(abs(memory_grid_fidelity(us,[0],BATH)[0]-exact))
        r=batch_pauli_expectations_from_choi(j[:1])[0];ex.append(r);ys.append(r.mean())
        pad=np.full((24,8),np.nan);pad[:l]=p;parameters.append(pad)
        fingerprints.append(hashlib.sha256(p.tobytes()).hexdigest())
        if (i+1)%512==0:print(family,'generate',i+1,flush=True)
    assert max(checks)<1e-10
    np.savez_compressed(dest,y=ys,ex=ex,features=features,length=lengths,eta=eta,parameters=parameters,
        x_fingerprints=fingerprints,product_affine=np.array(features)[:,1],exact_marginal_affine=np.array(features)[:,2],
        parameter_seed=parameter,eta_seed=eta_seed,max_reset_error=max(checks))

def evaluate(family,split,ckpts):
    design=json.loads(DESIGN.read_text());root=ROOT/family/split;root.mkdir(parents=True,exist_ok=True)
    if (root/'summary.csv').exists():raise FileExistsError(root)
    locked=verify_lock(family) if split=='test' else None
    if split=='validation' and (DATA/family/'test.npz').exists():raise RuntimeError('test already exists')
    cal=get_data(family,'calibration',ckpts);query=get_data(family,split,ckpts)
    query_set=set(query['x_fingerprints']);assert not (set(cal['x_fingerprints'])&query_set)
    if split=='test':
        for previous in ['validation']:
            assert not (query_set&set(get_data(family,previous,ckpts)['x_fingerprints']))
        oldpaths=([Path('data_revision/stage4/two_bath/test.npz')] if family=='two_bath' else [
            Path('results_revision/stage2')/family/'cache_test.npz',Path('results_revision/stage4')/family/'cache_test.npz'])
        for p in oldpaths:
            with np.load(p) as z:assert not(query_set&set(z['x_fingerprints']))
    names=['constant_mean','summary_ridge']+(['neural_'+Path(p).stem for p in ckpts] if family!='two_bath' else [])
    allocations=['fixed'] if family=='two_bath' else ['fixed','pilot']+(['zz_known'] if family=='zz' else [])
    budgets=design['two_bath_budgets'] if family=='two_bath' else design['budgets']
    configs=None
    if locked:
        configs={(c['method'],c['allocation'],c['shots']) for r in locked['selections'] if (c:=r['configuration'])}
        diagnostic=([20,36] if family=='zz' else [3,4,6,8,12,20,36] if family=='exchange' else [4,8,12,20,36])
        alloc='zz_known' if family=='zz' else 'fixed'
        for b in diagnostic:
            configs.add(('dfe',alloc,b))
            for name in names:
                for suffix in ['_iv','_eb']:configs.add((canonical(name)+suffix,alloc,b))
    rows=[];aggregates={};errors=[];ids=[];moments=[];n=len(query['y']);allq=np.arange(n)
    def add(name,alloc,b,draw,rep,pred,cost):
        e=np.abs(np.clip(pred,0,1)-query['y']);method=canonical(name);key=(method,alloc,b)
        rows.append(dict(method=method,allocation=alloc,shots=b,cal_draw=draw,repeat=rep,
            checkpoint=name if name.startswith('neural_') else 'none',mae=float(e.mean()),offline_shots=cost))
        if key not in aggregates:aggregates[key]=[np.zeros(n),0]
        aggregates[key][0]+=e;aggregates[key][1]+=1
        if split=='test':errors.append(e.astype(np.float32));ids.append(len(rows)-1)
    def wanted(m,a,b):return configs is None or (m,a,b) in configs
    seed={'zz':202610101,'exchange':202610102,'two_bath':202610055}[family]
    for draw in range(20):
        idx=np.arange(draw*64,(draw+1)*64)
        for repeat in range(5):
            labels,lv,_,observations=sample_measurement(cal['ex'][idx],64,
                np.random.default_rng(np.random.SeedSequence([seed,draw,repeat,1])),
                'zz_known' if family=='zz' else 'fixed',return_observations=True)
            vcal=calibration_outcome_variances(observations)
            moments.append(dict(cal_draw=draw,repeat=repeat,variance=vcal.tolist()))
            priors={}
            for name in names:
                oof=np.zeros(64)
                for fold in range(4):
                    held=np.arange(64)%4==fold
                    oof[held]=fit(name,cal,cal,idx[~held],labels[~held],idx[held])
                p=fit(name,cal,query,idx,labels,allq)
                priors[name]=(p,residual_prior_variance(oof,labels,lv))
                add(name+'_prior','none',0,draw,repeat,p,4096)
            for ai,alloc in enumerate(allocations):
                for b in budgets:
                    if not any(wanted(m,alloc,b) for m in ['dfe']+[canonical(x)+s for x in names for s in ['_iv','_eb']]):continue
                    if split=='validation':
                        stream=([seed,draw,repeat,b,2] if family=='two_bath' else [seed,draw,repeat,ai,b,3])
                    else:stream=[design['test_measurement_seeds'][family],draw,repeat,ai,b,4]
                    obs,var,counts,details=sample_measurement(query['ex'],b,np.random.default_rng(np.random.SeedSequence(stream)),alloc,return_observations=True)
                    assert np.all(counts==b)
                    # ZZ calibration uses X/Y only; deterministic Z has variance zero.
                    settings=vcal if alloc=='zz_known' or family!='zz' else np.r_[vcal,0.]
                    iv=independent_measurement_variance(settings,details['setting_shots'])
                    if wanted('dfe',alloc,b):add('dfe',alloc,b,draw,repeat,obs,0)
                    for name,(p,tau) in priors.items():
                        for suffix,v in [('_iv',iv),('_eb',var)]:
                            if wanted(canonical(name)+suffix,alloc,b):
                                add(name+suffix,alloc,b,draw,repeat,empirical_bayes_fusion(p,obs,v,tau),4096)
        print(family,split,'group',draw+1,flush=True)
    df=pd.DataFrame(rows);df.to_csv(root/'runs.csv',index=False);summary=[]
    for (m,a,b),(summed,count) in aggregates.items():
        sub=df[(df.method==m)&(df.allocation==a)&(df.shots==b)];qm=summed/count
        dm=sub.groupby('cal_draw').mae.mean().values;sm=sub.groupby('checkpoint').mae.mean().values
        v=qm.var(ddof=1)/n
        if len(dm)>1:v+=dm.var(ddof=1)/len(dm)
        if len(sm)>1:v+=sm.var(ddof=1)/len(sm)
        summary.append(dict(method=m,allocation=a,shots=b,mae=float(qm.mean()),
            selection_score=float(qm.mean()+1.645*np.sqrt(v)),offline_shots=int(sub.offline_shots.max())))
    pd.DataFrame(summary).to_csv(root/'summary.csv',index=False)
    (root/'calibration_moments.json').write_text(json.dumps(moments))
    if split=='test':np.savez_compressed(root/'paired_errors.npz',absolute_errors=np.asarray(errors),row_ids=ids,truth=query['y'],lengths=query['length'])
    (root/'manifest.json').write_text(json.dumps(dict(design_sha256=digest(DESIGN),code_sha256=code_hashes(),
        selection_sha256=digest(ROOT/family/'selection.json') if locked else None,
        test_sha256=digest(DATA/family/'test.npz') if locked else None,n=n,calibration_n=len(cal['y'])),indent=2))
    if split=='validation':lock(family)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--family',choices=['zz','exchange','two_bath'],required=True)
    ap.add_argument('--mode',choices=['validation','generate','test'],required=True);ap.add_argument('--ckpts',nargs='*',default=[])
    a=ap.parse_args()
    if a.family!='two_bath' and len(a.ckpts)!=5:raise ValueError('exactly five frozen checkpoints required')
    generate(a.family) if a.mode=='generate' else evaluate(a.family,a.mode,a.ckpts)
