import numpy as np
from physics.baselines.choi_bounds import choi_trace_distance_lower_bound
from physics.channels.base import Channel
from physics.fidelity import entanglement_fidelity
from physics.channels.single_qubit import amplitude_damping, depolarizing


def test_identity_and_depolarising_normalisation():
    for p in [0., .1, .6, 1.]:
        ch = depolarizing(p)
        np.testing.assert_allclose(choi_trace_distance_lower_bound(ch.choi), entanglement_fidelity(ch), atol=1e-12)


def test_bound_holds_for_amplitude_damping_and_coherent_channels():
    for gamma in [.01, .4, .8]:
        ch = amplitude_damping(gamma)
        assert choi_trace_distance_lower_bound(ch.choi) <= entanglement_fidelity(ch) + 1e-12
    for theta in [.01, .2, 1.]:
        ch = Channel("rotation", 2, kraus=[np.diag([np.exp(-.5j*theta), np.exp(.5j*theta)])])
        assert choi_trace_distance_lower_bound(ch.choi) <= entanglement_fidelity(ch) + 1e-12


def test_general_pure_target_trace_distance_normalisation():
    u = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    v = u.reshape(-1, order="F")
    choi = np.outer(v, v.conj())
    np.testing.assert_allclose(choi_trace_distance_lower_bound(choi, u), 1, atol=1e-12)
