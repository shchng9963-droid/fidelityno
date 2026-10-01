from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

LEVELS=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

def metrics(pred,y,length,split,model,latency_ms=0.0):
    rows=[]
    q=np.repeat(pred[:,None],len(LEVELS),axis=1)
    cov=(y[:,None] <= q).mean(0); ece=float(np.abs(cov-LEVELS).mean())
    e=y[:,None]-q; pin=float(np.maximum(LEVELS[None,:]*e,(LEVELS[None,:]-1)*e).mean())
    for L in sorted(set(length.tolist())):
        idx=length==L
        rows.append({'model':model,'seed':0,'split':split,'length':int(L),'mae':float(np.abs(pred[idx]-y[idx]).mean()),'pinball':pin,'crps':2*pin,'ece':ece,'latency_ms':latency_ms})
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='data'); ap.add_argument('--out',default='results/analytic.csv'); args=ap.parse_args()
    rows=[]
    for split in ['id_test','length_ood','family_ood']:
        d=np.load(Path(args.data_dir)/f'{split}.npz', allow_pickle=True); y=d['y']; length=d['length']; pf=d['per_fid']; mask=d['mask']
        prod=np.prod(np.where(mask>0,pf,1.0),axis=1)
        # Historical infidelity-sum proxy. It is not a general FvG bound.
        fvg=np.clip(1.0-np.sum(np.where(mask>0,1-pf,0.0),axis=1),0,1)
        # Retain historical numerical proxies under honest names. Neither
        # this square-root expression nor its maximum with the product is
        # a certified general diamond-norm or composed-fidelity bound.
        sqrt_term = np.sqrt(np.clip(1 - pf**2, 0, None))
        diamond_lb = np.clip(1.0 - np.sum(np.where(mask>0, sqrt_term, 0.0), axis=1), 0, 1)
        diamond_best = np.maximum(diamond_lb, prod)
        rows += metrics(prod,y,length,split,'product_approximation')
        rows += metrics(fvg,y,length,split,'infidelity_sum_heuristic')
        rows += metrics(diamond_lb,y,length,split,'root_infidelity_proxy')
        rows += metrics(diamond_best,y,length,split,'maximum_of_proxies')
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(args.out,index=False); print(f'wrote {args.out}')
if __name__=='__main__': main()
