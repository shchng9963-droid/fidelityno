
import numpy as np
from physics.channels.single_qubit import amplitude_damping, depolarizing, phase_damping
from physics.composition import compose_channels, process_fidelity

def test_choi_composition_associativity():
    a,b,c=amplitude_damping(0.1), depolarizing(0.05), phase_damping(0.2)
    left=compose_channels([compose_channels([a,b]),c])
    right=compose_channels([a,compose_channels([b,c])])
    assert np.allclose(left.superop, right.superop, atol=1e-10)
    assert abs(process_fidelity(left)-process_fidelity(right)) < 1e-10
