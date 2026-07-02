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
        # FvG bound: 1 - F_total >= max_i (1 - F_i)  -> F_total <= min_i F_i
        fvg=np.clip(1.0-np.sum(np.where(mask>0,1-pf,0.0),axis=1),0,1)
        # Diamond-norm telescope. Using FvG: 1/2 ||Lambda_i - I||_diam <= sqrt(1 - F_i^2) (Fuchs-van de Graaf upper bound).
        # Triangle: 1 - F_total <= sum_i (1/2 ||Lambda_i - I||_diam) <= sum_i sqrt(1 - F_i^2).
        # That gives a (potentially) tighter UB on fidelity than the additive infidelity bound when F_i close to 1.
        # We report the resulting LOWER bound on F: F_diamond_LB = max(0, 1 - sum sqrt(1 - F_i^2)).
        # As a "best-of" tighter bound we also report min(F_diamond_LB, F_product) which is the standard practice when reporting the tightest known analytic bound.
        sqrt_term = np.sqrt(np.clip(1 - pf**2, 0, None))
        diamond_lb = np.clip(1.0 - np.sum(np.where(mask>0, sqrt_term, 0.0), axis=1), 0, 1)
        diamond_best = np.maximum(diamond_lb, prod)  # take tighter (higher) of the two analytic LBs
        rows += metrics(prod,y,length,split,'product_bound')
        rows += metrics(fvg,y,length,split,'fvg_bound')
        rows += metrics(diamond_lb,y,length,split,'diamond_telescope')
        rows += metrics(diamond_best,y,length,split,'analytic_best')
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(args.out,index=False); print(f'wrote {args.out}')
if __name__=='__main__': main()
