import numpy as np
import pytest

from physics.baselines.dfe import direct_fidelity_estimate
from physics.baselines.hybrid import sample_identity_dfe, sample_identity_dfe_readout
from physics.baselines.revision_measurement import sample_measurement, residual_prior_variance
from physics.channels.single_qubit import depolarizing


def test_identity_requires_only_three_measured_settings():
    r = direct_fidelity_estimate([depolarizing(0)], strategy="stratified", total_shots=3)
    assert r.F_hat == 1
    assert r.quantum_shots == 3
    assert r.n_unique_paulis == 3


def test_nonidentity_target_is_not_silently_miscomputed():
    with pytest.raises(NotImplementedError):
        direct_fidelity_estimate([depolarizing(0)], target_unitary=np.diag([1, -1]))


def test_mean_and_variance_agree_with_binomial_law():
    ex = np.tile([1., .2, -.3, .4], (50000, 1))
    actual, sd = sample_identity_dfe(ex, 30, np.random.default_rng(71))
    truth = ex[0].mean()
    variance = np.sum(1 - ex[0, 1:]**2) / 10 / 16
    assert abs(actual.mean() - truth) < 5 * np.sqrt(variance / len(ex))
    assert abs(actual.var() / variance - 1) < .025
    np.testing.assert_allclose(sd**2, variance)


def test_identity_does_not_acquire_readout_noise():
    ex = np.tile([1., 0., 0., 0.], (20000, 1))
    raw, mitigated, _, _ = sample_identity_dfe_readout(ex, 96, .2, np.random.default_rng(77))
    assert abs(raw.mean() - .25) < .002
    assert abs(mitigated.mean() - .25) < .003


@pytest.mark.parametrize("allocation", ["fixed", "pilot", "zz_known"])
def test_deployable_measurement_unbiased_and_budgeted(allocation):
    ex = np.tile([1., .3, .3, 1.], (20000, 1))
    estimate, variance, counted = sample_measurement(ex, 32, np.random.default_rng(19), allocation)
    assert np.all(counted == 32)
    assert np.all(np.isfinite(variance)) and np.all(variance >= 0)
    assert abs(estimate.mean() - ex[0].mean()) < .003


def test_eb_subtracts_label_noise():
    assert residual_prior_variance(np.array([.5, .5]), np.array([.4, .6]), np.array([.02, .02])) == 0


def test_coherent_counterexample_to_infidelity_union_bound():
    theta = .01
    lengths = np.array([2, 4, 8, 16])
    epsilon = np.sin(theta / 2)**2
    true = np.cos(lengths * theta / 2)**2
    product = (1 - epsilon)**lengths
    ratio = (product - true) / (lengths * (lengths - 1) * epsilon)
    np.testing.assert_allclose(ratio, 1, atol=.003)
    assert np.all(1 - lengths * epsilon > true)
