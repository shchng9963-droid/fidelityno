"""Server-side implementation and provenance checks; does not change old results."""
import ast,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scripts.eval_revision_confirmatory import fit,digest
from physics.baselines.revision_measurement import (sample_measurement,
    calibration_outcome_variances,independent_measurement_variance,
    residual_prior_variance,empirical_bayes_fusion)

root=Path('results_revision/additional_priors_20261017')
report={'checkpoint_checks':[],'guard_checks':[],'regression_checks':{}}
with np.load('results_revision/stage2/zz/cache_calibration.npz') as z:cal=dict(z)
with np.load('results_revision/stage2/zz/cache_validation.npz') as z:val=dict(z)
provenance=json.loads(str(cal['provenance']))
for entry in json.loads(Path('results_revision/stage3/zz/checkpoints.json').read_text()):
    path=next(Path(p) for p in provenance['checkpoint_sha256'] if Path(p).stem==entry['checkpoint'])
    assert digest(path)==entry['sha256']==provenance['checkpoint_sha256'][str(path)]
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    cfg=ast.literal_eval(entry['cfg'])
    actual=checkpoint.get('cfg',checkpoint.get('config'))
    assert actual==cfg
    from train import make_model
    from omegaconf import OmegaConf
    model=make_model(cfg['model']['name'],32,48,OmegaConf.create(cfg))
    state=checkpoint.get('model',checkpoint.get('state_dict',checkpoint.get('model_state_dict')))
    model.load_state_dict(state)
    assert model.backbone.causal is False
    report['checkpoint_checks'].append({'name':entry['checkpoint'],'sha256':digest(path),
        'stored_causal_flag':cfg['model']['causal'],'effective_causal':model.backbone.causal,
        'source_data':cfg['data'],'curriculum':cfg['train']['curriculum']})

names=['summary_ridge']+['neural_'+e['checkpoint'] for e in json.loads(Path('results_revision/stage3/zz/checkpoints.json').read_text())]
errors={'ridge':[],'neural':[]};rows={'ridge':[],'neural':[]}
for g in range(20):
    idx=np.arange(g*64,(g+1)*64)
    for r in range(5):
        labels,lv,_,obs=sample_measurement(cal['ex'][idx],64,np.random.default_rng(np.random.SeedSequence([202610101,g,r,1])),
            'zz_known',return_observations=True)
        vcal=calibration_outcome_variances(obs)
        query,_,_,detail=sample_measurement(val['ex'],16,np.random.default_rng(np.random.SeedSequence([202610101,g,r,2,16,3])),
            'zz_known',return_observations=True)
        variance=independent_measurement_variance(vcal,detail['setting_shots'])
        for name in names:
            oof=np.zeros(64)
            for fold in range(4):
                held=np.arange(64)%4==fold
                oof[held]=fit(name,cal,cal,idx[~held],labels[~held],idx[held])
            prior=fit(name,cal,val,idx,labels,np.arange(4096))
            tau=residual_prior_variance(oof,labels,lv)
            pred=empirical_bayes_fusion(prior,query,variance,tau)
            e=np.abs(np.clip(pred,0,1)-val['y'])
            role='neural' if name.startswith('neural') else 'ridge'
            errors[role].append(e);rows[role].append(dict(g=g,r=r,s=name,mae=e.mean()))
    print('guard audit group',g+1,flush=True)
original=pd.read_csv('results_revision/stage5/zz/validation/summary.csv')
for role in errors:
    e=np.asarray(errors[role]);df=pd.DataFrame(rows[role]);terms={
        'query':float(e.mean(0).var(ddof=1)/4096),
        'group':float(df.groupby('g').mae.mean().var(ddof=1)/20),
        'checkpoint':float(df.groupby('s').mae.mean().var(ddof=1)/5) if role=='neural' else 0.}
    mean=float(e.mean());guard=mean+1.645*np.sqrt(sum(terms.values()))
    m='neural_iv' if role=='neural' else 'summary_ridge_iv'
    old=original[(original.method==m)&(original.allocation=='zz_known')&(original.shots==16)].iloc[0]
    assert abs(mean-old.mae)<1e-12 and abs(guard-old.selection_score)<1e-12
    report['guard_checks'].append(dict(role=role,shots=16,mean=mean,guard=float(guard),components=terms))
for family in ['zz','exchange']:
    for split in ['validation','test']:
        checks=json.loads((root/family/split/'regression_checks.json').read_text())
        assert checks and max(c['absolute_difference'] for c in checks)<1e-12
        report['regression_checks'][family+'/'+split]={'configurations':len(checks),
            'maximum_mae_difference':max(c['absolute_difference'] for c in checks)}
(root/'implementation_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
