from __future__ import annotations

import numpy as np

from physics.channels.base import choi_to_superop
from physics.channels.single_qubit import amplitude_damping
from physics.representations import choi_to_features
from scripts.eval_exact_composition import features_to_superop, infer_dim


def test_infer_dim_full_choi_features() -> None:
    assert infer_dim(32) == 2
    assert infer_dim(512) == 4


def test_vectorized_choi_to_superop_matches_reference() -> None:
    channel = amplitude_damping(0.13)
    features = choi_to_features(channel.choi, dim=2, mode="choi_hermitian")
    got = features_to_superop(features, dim=2)
    expected = choi_to_superop(channel.choi, d=2)
    np.testing.assert_allclose(got, expected, atol=1e-7)
