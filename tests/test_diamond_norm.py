"""Tests for the diamond-norm SDP baseline."""
from __future__ import annotations

import numpy as np
import pytest

cvxpy = pytest.importorskip("cvxpy")

from physics.baselines.diamond_norm import (
    diamond_norm_of_difference,
    fidelity_lower_bound_from_diamond,
)
from physics.channels.single_qubit import (
    amplitude_damping,
    depolarizing,
)
from physics.composition import compose_channels


def test_diamond_norm_identity_is_zero():
    """Diamond norm of (id - id) = 0 (within solver tolerance)."""
    # Use a near-identity channel: depolarizing with p=0
    ch = depolarizing(p=0.0)
    d = diamond_norm_of_difference(ch.choi, 2)
    assert abs(d) < 1e-3


def test_diamond_norm_depolarizing_matches_analytic():
    """For our depolarizing(p) convention (Pauli channel with rates
    p/3, p/3, p/3), the diamond distance to identity is 2*p
    (verified against qutip.dnorm)."""
    p = 0.1
    ch = depolarizing(p=p)
    d = diamond_norm_of_difference(ch.choi, 2)
    expected = 2 * p
    assert abs(d - expected) < 5e-3, f"diamond={d}, expected={expected}"


def test_diamond_norm_subadditive_under_composition():
    """||Lambda_1 . Lambda_2 - id||_diamond <= ||Lambda_1 - id||_diamond + ||Lambda_2 - id||_diamond."""
    chs = [depolarizing(p=0.05), amplitude_damping(gamma=0.07)]
    composed = compose_channels(chs)
    d_total = diamond_norm_of_difference(composed.choi, 2)
    d_sum = sum(diamond_norm_of_difference(c.choi, 2) for c in chs)
    assert d_total <= d_sum + 1e-3, f"d_total={d_total}, d_sum={d_sum}"


def test_fidelity_lower_bound_in_unit_interval():
    ch = depolarizing(p=0.2)
    d = diamond_norm_of_difference(ch.choi, 2)
    F_LB = fidelity_lower_bound_from_diamond(d)
    assert 0.0 <= F_LB <= 1.0
