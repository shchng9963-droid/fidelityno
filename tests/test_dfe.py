"""Unit tests for the DFE baseline."""
from __future__ import annotations

import numpy as np
import pytest

from physics.baselines.dfe import (
    chi_channel,
    chi_unitary,
    direct_fidelity_estimate,
)
from physics.channels.single_qubit import (
    amplitude_damping,
    depolarizing,
)
from physics.composition import compose_channels, exact_sequence_fidelity


def test_chi_unitary_identity_is_one() -> None:
    chi = chi_unitary(np.eye(2, dtype=complex), num_qubits=1)
    np.testing.assert_allclose(chi, np.ones(4))


def test_chi_channel_identity_recovers_one() -> None:
    # Identity (zero-strength depolarizing) channel should give chi = 1
    ch = depolarizing(p=0.0)
    chi = chi_channel(ch, num_qubits=1)
    np.testing.assert_allclose(chi, np.ones(4), atol=1e-10)


def test_chi_dot_target_recovers_F_e() -> None:
    # F_e(Lambda, V) = (1/d^2) sum_P chi_Lambda(P) chi_V(P)
    ch = amplitude_damping(gamma=0.07)
    composed = compose_channels([ch])
    chi_L = chi_channel(composed, num_qubits=1)
    chi_V = chi_unitary(np.eye(2, dtype=complex), num_qubits=1)
    F_recon = float(np.dot(chi_L, chi_V) / 2 ** 2)
    F_exact = exact_sequence_fidelity([ch])
    np.testing.assert_allclose(F_recon, F_exact, atol=1e-10)


def test_dfe_exact_mode_gives_zero_error_in_expectation() -> None:
    # With S=4^n=4 Paulis sampled WITH replacement against a uniform target,
    # the *expectation* of the importance-sampled estimator is F_exact.
    # Mean over many runs should converge.
    rng = np.random.default_rng(0)
    chs = [depolarizing(p=0.05), amplitude_damping(gamma=0.03)]
    F_exact = exact_sequence_fidelity(chs)
    estimates = [
        direct_fidelity_estimate(chs, num_paulis=20, noise="exact", rng=rng).F_hat
        for _ in range(200)
    ]
    np.testing.assert_allclose(np.mean(estimates), F_exact, atol=5e-3)


def test_dfe_finite_shot_std_decreases_with_M() -> None:
    rng = np.random.default_rng(0)
    chs = [depolarizing(p=0.02)]
    err_low = []
    err_high = []
    for _ in range(40):
        e_low = direct_fidelity_estimate(chs, num_paulis=20, M_per_pauli=10,
                                         noise="finite", rng=rng).abs_err
        e_high = direct_fidelity_estimate(chs, num_paulis=20, M_per_pauli=2000,
                                          noise="finite", rng=rng).abs_err
        err_low.append(e_low); err_high.append(e_high)
    # tighter shots should yield smaller average error
    assert np.mean(err_high) < np.mean(err_low)
