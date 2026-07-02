
import numpy as np
from physics.composition import channel_reference_fidelity

def predict_product_bound(seq):
    p=1.0
    for ch in seq: p*=channel_reference_fidelity(ch)
    return float(np.clip(p,0,1))

def predict_fvg_bound(seq):
    # Fuchs-van de Graaf motivated conservative lower proxy: infidelity union bound.
    inf=sum(max(0.0,1.0-channel_reference_fidelity(ch)) for ch in seq)
    return float(np.clip(1.0-inf,0,1))
