
import numpy as np
from physics.composition import process_fidelity, compose_channels

def predict_mc_kraus(seq, samples:int=100, seed:int=0):
    # Unbiased trajectory approximation for channels with Kraus lists. Falls back to exact if missing.
    rng=np.random.default_rng(seed); dim=seq[0].dim; acc=0.0
    psi=np.zeros((dim,),complex); psi[0]=1.0
    target=psi.copy()
    for _ in range(samples):
        state=psi.copy(); weight=1.0
        for ch in seq:
            if not ch.kraus: return process_fidelity(compose_channels(seq))
            probs=np.array([np.vdot(k@state,k@state).real for k in ch.kraus]); s=probs.sum()
            if s<=1e-12: break
            probs=probs/s; idx=rng.choice(len(ch.kraus),p=probs); state=ch.kraus[idx]@state/np.sqrt(max(probs[idx]*s,1e-12))
        acc += abs(np.vdot(target,state))**2
    return float(np.clip(acc/samples,0,1))
