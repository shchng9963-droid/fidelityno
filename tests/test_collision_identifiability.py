"""Guards for what is and is not identifiable from collision marginals."""
from __future__ import annotations

import numpy as np

from physics.channels.collision_nonmarkov import collision_sequence


def test_reset_bath_marginals_do_not_encode_eta() -> None:
    low = collision_sequence(12, eta=0.0, rng=np.random.default_rng(73))
    high = collision_sequence(12, eta=0.95, rng=np.random.default_rng(73))
    for marginal_low, marginal_high in zip(low.marginals, high.marginals):
        np.testing.assert_allclose(marginal_low.choi, marginal_high.choi, atol=0.0)
    assert abs(low.true_F_e - high.true_F_e) > 1e-3
