"""Cross-validate our F_e implementation against QuTiP's process_fidelity."""
import numpy as np
import pytest

qutip = pytest.importorskip('qutip')

from physics.channels.single_qubit import amplitude_damping, depolarizing, phase_damping
from physics.composition import compose_channels, process_fidelity


def test_process_fidelity_xval_qutip():
    """For Markovian sequences of single-qubit Pauli/AD/PD channels, our F_e
    must agree with qutip.process_fidelity to numerical precision.

    This is the cross-validation backing supplementary section S1.
    """
    rng = np.random.default_rng(42)
    n_trials = 256
    max_diff = 0.0

    for _ in range(n_trials):
        L = int(rng.choice([2, 4, 8, 16, 24]))
        seq = []
        for _ in range(L):
            which = rng.integers(0, 3)
            if which == 0:
                seq.append(amplitude_damping(0.01 + 0.3 * rng.random()))
            elif which == 1:
                seq.append(depolarizing(0.005 + 0.15 * rng.random()))
            else:
                seq.append(phase_damping(0.01 + 0.3 * rng.random()))
        f_ours = process_fidelity(compose_channels(seq))

        sop = None
        for ch in seq:
            s = qutip.Qobj(
                ch.superop,
                dims=[[[ch.dim], [ch.dim]], [[ch.dim], [ch.dim]]],
                superrep='super',
            )
            sop = s if sop is None else s * sop
        target = qutip.to_super(qutip.qeye(2))
        f_qutip = qutip.process_fidelity(sop, target)
        max_diff = max(max_diff, abs(f_ours - f_qutip))

    assert max_diff < 1e-7, f'max F_e diff vs qutip = {max_diff:.3e}'
