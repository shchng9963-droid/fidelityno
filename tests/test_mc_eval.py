import numpy as np
from physics.channels.single_qubit import amplitude_damping, depolarizing
from physics.channels.two_qubit import cnot_unitary
from physics.composition import compose_channels, process_fidelity
from scripts.eval_mc import kraus_from_choi, exact_process_fidelity_from_kraus, features_to_choi
from physics.channels.base import choi_to_real_features


def test_kraus_from_choi_recovers_process_fidelity():
    seq = [amplitude_damping(0.11), depolarizing(0.07)]
    comp = compose_channels(seq)
    recovered = kraus_from_choi(comp.choi, comp.dim)
    exact = exact_process_fidelity_from_kraus([recovered], comp.dim)
    assert abs(exact - process_fidelity(comp)) < 1e-8


def test_features_to_choi_roundtrip():
    ch = amplitude_damping(0.2)
    feat = choi_to_real_features(ch.choi)
    out = features_to_choi(feat)
    assert np.allclose(out, ch.choi, atol=1e-8)


def test_exact_process_fidelity_from_kraus_respects_nonidentity_target():
    unitary = cnot_unitary()
    kraus = [[unitary]]
    identity_target = exact_process_fidelity_from_kraus(kraus, 4)
    target_fidelity = exact_process_fidelity_from_kraus(kraus, 4, target_unitary=unitary)
    assert identity_target == 0.25
    assert abs(target_fidelity - 1.0) < 1e-8
