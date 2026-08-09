import numpy as np

from physics.baselines.dfe import chi_channel
from physics.baselines.hybrid import (
    allocate_two_level_budget,
    ambiguity_statistics,
    apply_budgeted_convex_fusion,
    apply_convex_fusion,
    batch_pauli_expectations_from_choi,
    complete_identity_dfe,
    fit_budgeted_convex_fusion,
    fit_convex_fusion,
    sample_identity_dfe,
    sample_identity_dfe_readout,
    sample_identity_dfe_pilot,
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


def test_two_stage_identity_dfe_reuses_pilot_and_respects_budget() -> None:
    expectations = np.ones((10, 4))
    pilot_plus, pilot = sample_identity_dfe_pilot(
        expectations, total_shots=8, rng=np.random.default_rng(8)
    )
    budgets = np.array([16] * 5 + [48] * 5)
    final, sigma = complete_identity_dfe(
        expectations,
        pilot_plus,
        pilot_shots=8,
        final_total_shots=budgets,
        rng=np.random.default_rng(9),
    )
    assert np.allclose(pilot, 1.0)
    assert np.allclose(final, 1.0)
    assert np.allclose(sigma, 0.0)


def test_two_level_allocation_and_budgeted_fusion() -> None:
    scores = np.arange(12, dtype=float)
    budgets = allocate_two_level_budget(scores, 16, 48)
    assert np.array_equal(budgets[:6], np.full(6, 16))
    assert np.array_equal(budgets[6:], np.full(6, 48))
    prior = np.linspace(0.1, 0.8, 12)
    measurement = np.linspace(0.9, 0.2, 12)
    true_weights = {16: 0.25, 48: 0.75}
    target = np.where(
        budgets == 16,
        apply_convex_fusion(prior, measurement, true_weights[16]),
        apply_convex_fusion(prior, measurement, true_weights[48]),
    )
    fitted = fit_budgeted_convex_fusion(prior, measurement, target, budgets)
    prediction = apply_budgeted_convex_fusion(prior, measurement, budgets, fitted)
    assert np.isclose(fitted[16], true_weights[16])
    assert np.isclose(fitted[48], true_weights[48])
    assert np.allclose(prediction, target)


def test_readout_noise_zero_matches_standard_dfe() -> None:
    expectations = np.array([[1.0, 0.4, -0.2, 0.7], [1.0, -0.1, 0.3, 0.2]])
    standard, _ = sample_identity_dfe(
        expectations, 64, np.random.default_rng(19)
    )
    raw, mitigated, raw_sigma, mitigated_sigma = sample_identity_dfe_readout(
        expectations, 64, 0.0, np.random.default_rng(19)
    )
    np.testing.assert_allclose(raw, standard)
    np.testing.assert_allclose(mitigated, standard)
    np.testing.assert_allclose(raw_sigma, mitigated_sigma)


def test_readout_mitigation_reduces_large_shot_bias() -> None:
    expectations = np.tile(np.array([1.0, 0.8, 0.6, 0.4]), (200, 1))
    truth = expectations.mean(axis=1)
    raw, mitigated, _, _ = sample_identity_dfe_readout(
        expectations, 4000, 0.08, np.random.default_rng(23)
    )
    assert np.abs(mitigated - truth).mean() < np.abs(raw - truth).mean()
