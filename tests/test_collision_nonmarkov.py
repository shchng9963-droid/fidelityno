"""Tests for the non-Markovian collision-model channel family."""
from __future__ import annotations

import numpy as np
import pytest

from physics.channels.collision_nonmarkov import (
    _collision_unitary,
    _marginal_choi_from_unitary,
    _system_choi_from_joint_propagation,
    collision_sequence,
)
from physics.channels.base import Channel, partial_trace_output_choi
from physics.composition import compose_channels, exact_sequence_fidelity


def _is_cptp(choi: np.ndarray, dim: int) -> bool:
    # Hermitian
    if not np.allclose(choi, choi.conj().T, atol=1e-9):
        return False
    # PSD
    evals = np.linalg.eigvalsh(0.5 * (choi + choi.conj().T))
    if evals.min() < -1e-8:
        return False
    # Trace-preserving: partial trace over output = I
    pt = partial_trace_output_choi(choi, dim)
    return np.allclose(pt, np.eye(dim), atol=1e-7)


def test_collision_unitary_is_unitary() -> None:
    U = _collision_unitary(J=0.1, omega=0.2, tau=0.5)
    np.testing.assert_allclose(U @ U.conj().T, np.eye(4), atol=1e-10)


def test_marginal_is_cptp() -> None:
    U = _collision_unitary(J=0.1, omega=0.2, tau=0.7)
    rho_B = 0.5 * np.eye(2, dtype=complex)
    C = _marginal_choi_from_unitary(U, rho_B)
    assert _is_cptp(C, dim=2)


def test_eta_zero_recovers_markovian() -> None:
    # eta=0 forces bath reset to |+> every collision; then the true
    # composed channel equals the product of marginals.
    rng = np.random.default_rng(1)
    sample = collision_sequence(num_collisions=4, eta=0.0, rng=rng)
    F_marg = exact_sequence_fidelity(sample.marginals)
    np.testing.assert_allclose(sample.true_F_e, F_marg, atol=1e-9)


def test_eta_one_is_non_markovian() -> None:
    rng = np.random.default_rng(2)
    sample = collision_sequence(num_collisions=8, eta=1.0, rng=rng)
    F_marg = exact_sequence_fidelity(sample.marginals)
    # When the bath is fully retained, the marginal product cannot
    # generally equal the true value -- otherwise the family would be
    # trivially Markovian.
    assert abs(sample.true_F_e - F_marg) > 1e-4


def test_marginals_are_each_cptp() -> None:
    rng = np.random.default_rng(3)
    sample = collision_sequence(num_collisions=6, eta=0.7, rng=rng)
    for m in sample.marginals:
        assert _is_cptp(m.choi, dim=2)


def test_true_channel_is_cptp() -> None:
    rng = np.random.default_rng(4)
    Us = [_collision_unitary(0.1, 0.2, 0.5) for _ in range(5)]
    C = _system_choi_from_joint_propagation(Us, eta=0.5)
    assert _is_cptp(C, dim=2)


def test_product_bound_fails_for_eta_high() -> None:
    """The headline physics: at eta close to 1, |F_marg - F_true| should
    grow with sequence length, in contrast to eta=0 where it stays 0.

    We compare average gap at length 16 vs length 4.
    """
    gaps_short = []
    gaps_long = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        s4 = collision_sequence(num_collisions=4, eta=0.95, rng=rng)
        s16 = collision_sequence(num_collisions=16, eta=0.95, rng=rng)
        gaps_short.append(abs(s4.true_F_e - exact_sequence_fidelity(s4.marginals)))
        gaps_long.append(abs(s16.true_F_e - exact_sequence_fidelity(s16.marginals)))
    assert np.mean(gaps_long) > 1.2 * np.mean(gaps_short)
