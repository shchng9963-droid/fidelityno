import numpy as np
import pytest
from physics.channels.batched_memory import memory_grid_choi
from physics.channels.collision_nonmarkov import _collision_unitary, PLUS, _system_choi_from_joint_propagation
from physics.channels.collision_exchange_memory import _exchange_unitary, DEFAULT_BATH


@pytest.mark.parametrize("family", ["zz", "exchange"])
@pytest.mark.parametrize("length", [1, 8, 24])
def test_batched_propagation_matches_independent_operator_basis(family, length):
    rng = np.random.default_rng(23)
    us = ([_collision_unitary(*rng.uniform(.05,.4,3)) for _ in range(length)] if family=="zz"
          else [_exchange_unitary(*rng.uniform(.05,.4,4)) for _ in range(length)])
    bath = PLUS if family=="zz" else DEFAULT_BATH
    eta = np.array([0,.4,.85,.99,1])
    got = memory_grid_choi(us, eta, bath)
    expected = np.array([_system_choi_from_joint_propagation(us, float(e), bath) for e in eta])
    np.testing.assert_allclose(got, expected, atol=2e-13)
    np.testing.assert_allclose(np.trace(got,axis1=1,axis2=2), 2, atol=2e-13)
