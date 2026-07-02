
import numpy as np
import pytest
from physics.channels.single_qubit import amplitude_damping, phase_damping, depolarizing, pauli_channel
from physics.channels.two_qubit import correlated_dephasing, two_qubit_depolarizing, imperfect_gate
from physics.channels.lindblad import sample_lindblad

def test_single_qubit_channels_are_cptp():
    chans=[amplitude_damping(0.1), phase_damping(0.2), depolarizing(0.15), pauli_channel(0.02,0.03,0.04)]
    assert all(ch.is_cptp(1e-8) for ch in chans)

def test_two_qubit_channels_are_cptp():
    chans=[correlated_dephasing(0.1,0.7), two_qubit_depolarizing(0.05), imperfect_gate('cnot',0.02,0.01), imperfect_gate('swap',-0.01,0.02)]
    assert all(ch.is_cptp(1e-8) for ch in chans)

def test_lindblad_sample_is_cptp():
    ch=sample_lindblad(np.random.default_rng(0))
    assert ch.is_cptp(1e-7)
