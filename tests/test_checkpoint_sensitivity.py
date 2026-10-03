"""Tests for the exploratory sensitivity helper; no primary analysis changes."""
import numpy as np
from scripts.analyze_revision_checkpoint_sensitivity import resampling_weights, bootstrap


def test_resampling_weights_normalized_and_repeatable():
    first = resampling_weights(11, seed=123, nboot=17)
    second = resampling_weights(11, seed=123, nboot=17)
    for x, y in zip(first, second):
        np.testing.assert_array_equal(x, y)
        np.testing.assert_allclose(x.sum(1), 1, atol=2e-7)
    assert first[0].shape == (17, 100)
    assert first[1].shape == (17, 400)
    assert first[2].shape == (17, 11)


def test_constant_bootstrap_preserves_paired_difference():
    w, ws, qw = resampling_weights(11, seed=321, nboot=17)
    a = np.full((20, 5, 4, 11), .3, dtype=np.float32)
    b = np.full((20, 5, 1, 11), .2, dtype=np.float32)
    np.testing.assert_allclose(bootstrap(a, ws, qw) - bootstrap(b, w, qw), .1, atol=3e-7)


def test_leave_one_out_mean_identity():
    x = np.random.default_rng(123).random((20, 5, 5, 11))
    checkpoint = x.mean(axis=(0, 1, 3))
    for omitted in range(5):
        kept = [s for s in range(5) if s != omitted]
        np.testing.assert_allclose(x[:, :, kept, :].mean(),
                                   (5 * x.mean() - checkpoint[omitted]) / 4)
