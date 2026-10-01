"""Numerical polynomial extrema of fixed-input retention trajectories."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from numpy.polynomial import Chebyshev
from scipy.optimize import minimize_scalar
from physics.channels.batched_memory import memory_grid_fidelity
from physics.channels.collision_nonmarkov import _collision_unitary,PLUS
from physics.channels.collision_exchange_memory import _exchange_unitary,DEFAULT_BATH

def extrema(us,bath,lo,hi):
    degree=len(us)-1
    nodes=np.cos(np.pi*(np.arange(degree+1)+.5)/(degree+1))
    eta=(lo+hi)/2+(hi-lo)/2*nodes
    f=memory_grid_fidelity(us,eta,bath)
    poly=Chebyshev.fit(eta,f,degree,domain=[lo,hi])
    roots=poly.deriv().roots()
    stationary=np.sort(roots.real[(abs(roots.imag)<1e-7)&(roots.real>lo)&(roots.real<hi)])
    points=np.r_[lo,stationary,hi]
    vals=memory_grid_fidelity(us,points,bath)
    check=np.linspace(lo,hi,2*degree+19)
    residual=float(np.max(abs(poly(check)-memory_grid_fidelity(us,check,bath))))
    if residual>2e-10: raise RuntimeError('polynomial interpolation failed')
    # Polish candidate minima and maxima using direct physical propagation.
    for sign in [1,-1]:
        j=int(np.argmin(sign*vals))
        if 0<j<len(points)-1:
            opt=minimize_scalar(lambda t:sign*memory_grid_fidelity(us,[t],bath)[0],
                bounds=((points[j-1]+points[j])/2,(points[j]+points[j+1])/2),method='bounded',
                options={'xatol':1e-13})
            if opt.success:
                points=np.r_[points,opt.x]; vals=np.r_[vals,sign*opt.fun]
                order=np.argsort(points);points=points[order];vals=vals[order]
    jlo,jhi=int(np.argmin(vals)),int(np.argmax(vals))
    derivative=poly.deriv()(np.linspace(lo,hi,513))
    return dict(minimum=float(vals[jlo]),maximum=float(vals[jhi]),span=float(np.ptp(vals)),
        eta_min=float(points[jlo]),eta_max=float(points[jhi]),stationary_count=len(stationary),
        polynomial_check_error=residual,derivative_changes_sign=bool(derivative.min()<-1e-9 and derivative.max()>1e-9))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',required=True,choices=['zz','exchange']);a=ap.parse_args()
    raw=np.load(Path('results_revision/stage2')/a.family/'grid/audit.npz')
    make,bath=(_collision_unitary,PLUS) if a.family=='zz' else (_exchange_unitary,DEFAULT_BATH)
    rows=[]
    for i,(p,l) in enumerate(zip(raw['parameters'],raw['lengths'])):
        us=[make(*x) for x in p[:l]]
        for lo,hi in [(.85,.99),(0.,1.)]:
            result=extrema(us,bath,lo,hi)
            result.update(base=i,length=int(l),eta_low=lo,eta_high=hi,
                grid113_span=float(np.ptp(raw['fidelity_grid'][i])))
            rows.append(result)
        if (i+1)%256==0:print(a.family,'extrema',i+1,flush=True)
    root=Path('results_revision/stage4')/a.family/'extrema';root.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows);df.to_csv(root/'per_base.csv',index=False)
    summary=df.groupby(['eta_low','eta_high']).agg(mean_span=('span','mean'),
        maximum_polynomial_error=('polynomial_check_error','max'),
        nonmonotone_fraction=('derivative_changes_sign','mean')).reset_index()
    summary.to_csv(root/'summary.csv',index=False); print(summary.to_string(index=False),flush=True)
    (root/'manifest.json').write_text(json.dumps(dict(scope='one fixed-parameter eta trajectory, not full representation fibre',
        method='degree L-1 Chebyshev fit; all real derivative roots and endpoints; physical checks and local bounded polishing',
        certificate='numerical double-precision extrema, no formal global certificate'),indent=2))
if __name__=='__main__':main()
