
import numpy as np
from physics.composition import channel_reference_fidelity

def predict_product_bound(seq):
    p=1.0
    for ch in seq: p*=channel_reference_fidelity(ch)
    return float(np.clip(p,0,1))

def predict_infidelity_sum_heuristic(seq):
    """Additive infidelity proxy; not a lower bound for general channels."""
    inf=sum(max(0.0,1.0-channel_reference_fidelity(ch)) for ch in seq)
    return float(np.clip(1.0-inf,0,1))


def predict_fvg_bound(seq):
    """Legacy entry point. Use an explicitly named estimator in new work."""
    import warnings
    warnings.warn("This legacy estimator is an infidelity-sum heuristic, not a general FvG bound",
                  DeprecationWarning, stacklevel=2)
    return predict_infidelity_sum_heuristic(seq)
