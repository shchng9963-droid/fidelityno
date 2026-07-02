"""Randomised composition associativity test (run with `pytest tests/test_composition_random.py`)."""
import numpy as np
import pytest
from physics.channels.single_qubit import amplitude_damping, depolarizing, phase_damping
from physics.channels.two_qubit import correlated_dephasing, two_qubit_depolarizing
from physics.composition import compose_channels, process_fidelity


_SQ_CHOICES = [
    lambda r: amplitude_damping(0.05 + 0.4 * r.random()),
    lambda r: depolarizing(0.005 + 0.2 * r.random()),
    lambda r: phase_damping(0.01 + 0.3 * r.random()),
]
_TQ_CHOICES = [
    lambda r: correlated_dephasing(0.005 + 0.15 * r.random()),
    lambda r: two_qubit_depolarizing(0.005 + 0.15 * r.random()),
]


def _draw_triple(rng, dim):
    if dim == 2:
        choices = _SQ_CHOICES
    else:
        choices = _TQ_CHOICES
    return tuple(rng.choice(choices)(rng) for _ in range(3))


@pytest.mark.parametrize('dim', [2, 4])
def test_choi_composition_associativity_randomised(dim):
    rng = np.random.default_rng(20260612 + dim)
    n_trials = 1000  # 1k per dim → 2k total; quick
    max_super = 0.0
    max_fid = 0.0
    for _ in range(n_trials):
        a, b, c = _draw_triple(rng, dim)
        left = compose_channels([compose_channels([a, b]), c])
        right = compose_channels([a, compose_channels([b, c])])
        max_super = max(max_super, np.max(np.abs(left.superop - right.superop)))
        max_fid = max(max_fid, abs(process_fidelity(left) - process_fidelity(right)))
    assert max_super < 1e-10, f'super max diff {max_super}'
    assert max_fid < 1e-10, f'F_e max diff {max_fid}'
