"""Tests for the exchange-coupled hidden-memory collision family."""
from __future__ import annotations

import numpy as np

from physics.channels.collision_exchange_memory import (
    _exchange_hamiltonian,
    _exchange_unitary,
    exchange_collision_sequence,
    exchange_sequence_from_params,
)
from physics.composition import exact_sequence_fidelity


def test_exchange_terms_are_noncommuting() -> None:
    interaction = _exchange_hamiltonian(0.2, 0.0, 0.0)
    local = _exchange_hamiltonian(0.0, 0.0, 0.3)
    assert np.linalg.norm(interaction @ local - local @ interaction) > 1e-6


def test_exchange_unitary_is_unitary() -> None:
    unitary = _exchange_unitary(0.2, 0.05, 0.3, 0.7)
    np.testing.assert_allclose(unitary @ unitary.conj().T, np.eye(4), atol=1e-10)


def test_exchange_channels_are_cptp_and_eta_zero_composes() -> None:
    sample = exchange_collision_sequence(7, eta=0.0, rng=np.random.default_rng(5))
    assert all(channel.is_cptp(atol=1e-8) for channel in sample.marginals)
    assert sample.true_choi.shape == (4, 4)
    np.testing.assert_allclose(
        sample.true_F_e,
        exact_sequence_fidelity(sample.marginals),
        atol=1e-9,
    )


def test_exchange_marginals_do_not_reveal_eta() -> None:
    base = exchange_collision_sequence(8, eta=0.0, rng=np.random.default_rng(11))
    low = exchange_sequence_from_params(base.params, eta=0.2)
    high = exchange_sequence_from_params(base.params, eta=0.95)
    for left, right in zip(low.marginals, high.marginals):
        np.testing.assert_allclose(left.choi, right.choi, atol=1e-12)
    assert abs(low.true_F_e - high.true_F_e) > 1e-4
