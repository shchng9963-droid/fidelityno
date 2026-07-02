
from __future__ import annotations
import argparse, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
from train import make_model, prediction_to_quantiles, mean_from_prediction

def pinball_np(q,y,levels):
    e=y[:,None]-q; lev=np.asarray(levels)[None,:]; return np.maximum(lev*e,(lev-1)*e).mean()
def ece_quantile(q,y,levels):
    cov=(y[:,None] <= q).mean(0); return float(np.abs(cov-np.asarray(levels)).mean()), cov

def crps_from_quantiles(q,y,levels): return float(2*pinball_np(q,y,levels))
def load(path): d=np.load(path, allow_pickle=True); return d, TensorDataset(torch.tensor(d['x']).float(),torch.tensor(d['mask']).float(),torch.tensor(d['y']).float(),torch.tensor(d['stats']).float())
def eval_ckpt(ckpt_path,splits,out_csv):
    ck=torch.load(ckpt_path,map_location='cpu',weights_only=False); cfg=OmegaConf.create(ck['cfg']); rows=[]; levels=cfg.model.quantiles
    sample=np.load(next(iter(splits.values())), allow_pickle=True); input_dim=sample['x'].shape[-1]; max_len=sample['x'].shape[1]
    model=make_model(cfg.model.name,input_dim,max_len,cfg); model.load_state_dict(ck['model']); model.eval()
    for name,path in splits.items():
        raw,ds=load(path); preds=[]; ys=[]; t0=time.perf_counter(); nseq=0
        with torch.no_grad():
            for x,m,y,stats in DataLoader(ds,batch_size=256):
                pred,_=model(x,m); q=prediction_to_quantiles(pred, torch.tensor(levels, dtype=torch.float32)); preds.append(q.numpy()); ys.append(y.numpy()); nseq+=len(y)
        elapsed=time.perf_counter()-t0; q=np.concatenate(preds); y=np.concatenate(ys); mean=q.mean(1); ece,cov=ece_quantile(q,y,levels)
        for L in sorted(set(raw['length'].tolist())):
            idx=raw['length']==L; rows.append({'model':cfg.model.name,'head_type':cfg.model.get('head_type','quantile'),'seed':cfg.seed,'split':name,'length':int(L),'mae':float(np.abs(mean[idx]-y[idx]).mean()),'pinball':pinball_np(q[idx],y[idx],levels),'crps':crps_from_quantiles(q[idx],y[idx],levels),'ece':ece,'latency_ms':1000*elapsed/max(nseq,1)})
    df=pd.DataFrame(rows); Path(out_csv).parent.mkdir(parents=True,exist_ok=True); df.to_csv(out_csv,index=False); print(df)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--ckpt',required=True); ap.add_argument('--data-dir',default='data'); ap.add_argument('--out',default='results/summary.csv'); args=ap.parse_args()
    splits={k:str(Path(args.data_dir)/f'{k}.npz') for k in ['id_test','length_ood','family_ood']}; eval_ckpt(args.ckpt,splits,args.out)
