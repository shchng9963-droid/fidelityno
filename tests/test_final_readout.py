"""Known-rate variance propagation and unchanged measurement streams."""
from math import comb
import numpy as np
from physics.baselines.revision_measurement import (
    sample_measurement, calibration_outcome_variances, independent_measurement_variance,
)


def test_known_rate_variance_corrected_once():
    a = .9
    r = np.array([.2, .4, .6])
    counts = np.array([[10, 12, 14]])
    proxy = independent_measurement_variance(1 - (a * r)**2, counts, a)
    expected = np.sum((1 / a**2 - r**2) / counts, axis=1) / 16
    np.testing.assert_allclose(proxy, expected, atol=1e-15)


def test_calibration_variance_matches_clipped_unbiased_moment():
    m, a, r = 8, .9, .3
    probability = (1 + a * r) / 2
    expectation, raw_expectation = 0., 0.
    for k in range(m + 1):
        observation = dict(observed_means=np.full((1, 2), 2 * k / m - 1),
                           setting_shots=np.full((1, 2), m), allocation='zz_known')
        v = calibration_outcome_variances(observation)
        raw = m / (m - 1) * (1 - (2 * k / m - 1)**2)
        np.testing.assert_allclose(v, np.full(2, np.clip(raw, 0, 1)))
        mass = comb(m, k) * probability**k * (1 - probability)**(m - k)
        expectation += mass * v[0]
        raw_expectation += mass * raw
    np.testing.assert_allclose(raw_expectation, 1 - (a * r)**2, atol=1e-15)
    assert expectation <= raw_expectation


def test_readout_observation_return_preserves_stream_and_cost():
    ex = np.tile([1., .3, .3, 1.], (30, 1))
    old = sample_measurement(ex, 20, np.random.default_rng(21), 'zz_known', .05)
    new = sample_measurement(ex, 20, np.random.default_rng(21), 'zz_known', .05,
                             return_observations=True)
    for before, after in zip(old, new[:3]):
        np.testing.assert_array_equal(before, after)
    assert np.all(new[3]['setting_shots'].sum(axis=1) == 20)
    assert new[3]['attenuation'] == .9
