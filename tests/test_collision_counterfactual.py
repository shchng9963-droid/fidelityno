import numpy as np

from physics.channels.collision_counterfactual import (
    collision_fidelity_grid_from_params,
    collision_sequence_from_params,
)
from physics.channels.collision_nonmarkov import collision_sequence


def test_replay_changes_truth_but_not_marginals() -> None:
    base = collision_sequence(8, eta=0.2, rng=np.random.default_rng(7))
    low = collision_sequence_from_params(base.params, eta=0.0)
    high = collision_sequence_from_params(base.params, eta=0.95)
    for left, right in zip(low.marginals, high.marginals):
        assert np.allclose(left.choi, right.choi, atol=1e-13)
    assert abs(low.true_F_e - high.true_F_e) > 1e-4


def test_replay_matches_original_at_same_eta() -> None:
    base = collision_sequence(5, eta=0.63, rng=np.random.default_rng(13))
    replay = collision_sequence_from_params(base.params, eta=base.eta)
    assert np.allclose(base.true_choi, replay.true_choi, atol=1e-13)
    assert np.isclose(base.true_F_e, replay.true_F_e, atol=1e-13)


def test_grid_reuses_parameters_without_changing_values() -> None:
    base = collision_sequence(6, eta=0.4, rng=np.random.default_rng(23))
    eta = np.array([0.0, 0.3, 0.9])
    grid = collision_fidelity_grid_from_params(base.params, eta)
    direct = np.array(
        [collision_sequence_from_params(base.params, eta=float(value)).true_F_e for value in eta]
    )
    assert np.allclose(grid, direct, atol=1e-13)
