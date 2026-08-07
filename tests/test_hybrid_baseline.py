import numpy as np

from physics.baselines.dfe import chi_channel
from physics.baselines.hybrid import (
    ambiguity_statistics,
    apply_convex_fusion,
    batch_pauli_expectations_from_choi,
    fit_convex_fusion,
    sample_identity_dfe,
)
from physics.channels.single_qubit import depolarizing


def test_batched_pauli_expectations_match_channel_implementation() -> None:
    channel = depolarizing(0.07)
    expected = chi_channel(channel, 1)
    actual = batch_pauli_expectations_from_choi(channel.choi)[0]
    assert np.allclose(actual, expected, atol=1e-13)


def test_identity_dfe_is_exact_for_identity_channel() -> None:
    estimates, sigma = sample_identity_dfe(
        np.ones((12, 4)), total_shots=4, rng=np.random.default_rng(4)
    )
    assert np.allclose(estimates, 1.0)
    assert np.allclose(sigma, 0.0)


def test_convex_fusion_recovers_known_weight() -> None:
    prior = np.linspace(0.1, 0.8, 20)
    measurement = np.linspace(0.9, 0.2, 20)
    target = 0.7 * prior + 0.3 * measurement
    weight = fit_convex_fusion(prior, measurement, target)
    assert np.isclose(weight, 0.3)
    assert np.allclose(apply_convex_fusion(prior, measurement, weight), target)


def test_ambiguity_statistics_have_expected_bounds() -> None:
    values = np.array([[0.2, 0.5, 0.8], [0.4, 0.4, 0.6]])
    stats = ambiguity_statistics(values)
    assert np.allclose(stats["diameter"], [0.6, 0.2])
    assert np.allclose(stats["minimax_abs_lower_bound"], [0.3, 0.1])
    assert np.allclose(stats["conditional_median"], [0.5, 0.4])
