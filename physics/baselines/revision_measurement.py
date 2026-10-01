"""Deployable identity-target measurement and shrinkage baselines.

No exact conditional variance is exposed to a fitted estimator. The optional
ZZ allocation uses the specified dephasing family, not per-query truth.
"""
from __future__ import annotations

import numpy as np

from physics.baselines.dfe import _allocate_stratified_shots
from physics.baselines.hybrid import _identity_allocation


def sample_measurement(expectations, budget, rng, allocation="fixed", readout_error=0.0):
    """Return estimate, observable variance estimate, and counted shots.

    Pilot allocation uses its pilot only to choose independent stage-two
    measurements. This avoids a biased pooled mean from outcome-dependent
    per-setting sample sizes. The pilot cost remains in the total budget.
    A known symmetric readout rate is used only in controlled stress tests.
    """
    means, fixed = _identity_allocation(expectations, budget)
    if not 0 <= readout_error < 0.5:
        raise ValueError("invalid readout error")
    attenuation = 1 - 2 * readout_error
    n = len(means)
    offset, factor = 0.25, 0.25
    if allocation == "zz_known":
        # [U_t,Z_S]=0 and the bath reset preserves system populations,
        # so chi_Z=1. This assumption is not valid in the exchange model.
        if not np.allclose(means[:, 2], 1.0, atol=1e-7):
            raise ValueError("ZZ structural baseline requires chi_Z=1")
        means = means[:, :2]
        shots = np.broadcast_to(_allocate_stratified_shots(np.ones(2), budget)[1], means.shape)
        offset = 0.5
    elif allocation == "fixed":
        shots = np.broadcast_to(fixed, means.shape)
    elif allocation == "pilot":
        # Three shots per observable plus at least two independent final
        # observations per setting. Small budgets use fixed allocation.
        if budget < 15:
            shots = np.broadcast_to(fixed, means.shape)
        else:
            pilot_per = 3
            pilot_plus = rng.binomial(pilot_per, (1 + attenuation * means) / 2)
            p = (pilot_plus + 0.5) / (pilot_per + 1)
            sd = np.sqrt(4 * p * (1 - p))
            remaining = budget - 3 * pilot_per - 6
            expected = remaining * sd / sd.sum(axis=1, keepdims=True)
            extra = np.floor(expected).astype(int)
            shots = 2 + extra
            remainder = remaining - extra.sum(axis=1)
            order = np.argsort(-(expected - extra), axis=1, kind="stable")
            for k in range(3):
                ids = np.flatnonzero(remainder > k)
                shots[ids, order[ids, k]] += 1
    else:
        raise ValueError(allocation)
    plus = rng.binomial(shots, (1 + attenuation * means) / 2)
    obs = 2 * plus / shots - 1
    estimate = offset + factor * (obs / attenuation).sum(axis=1)
    # For n>1: (1-observed_mean**2)/(n-1) estimates Var(sample mean).
    # At n=1 use a conservative upper bound instead of an oracle variance.
    obs_var = np.where(shots > 1, (1 - obs**2) / np.maximum(shots - 1, 1), 1.0)
    estimated_variance = factor**2 * obs_var.sum(axis=1) / attenuation**2
    return estimate, estimated_variance, np.full(n, budget, dtype=int)


def residual_prior_variance(oof_prior, noisy_labels, label_variance):
    """Noise-corrected out-of-fold residual second moment, truncated at zero."""
    return float(max(0.0, np.mean((noisy_labels - oof_prior)**2 - label_variance)))


def empirical_bayes_fusion(prior, measurement, measurement_variance, prior_variance):
    """Normal-moment shrinkage sensitivity baseline, not a Bayes-MAE claim."""
    variance = np.maximum(np.asarray(measurement_variance), 0.0)
    weight = np.divide(prior_variance, prior_variance + variance,
                       out=np.ones_like(variance), where=(prior_variance + variance) > 0)
    return np.asarray(prior) + weight * (np.asarray(measurement) - prior)
