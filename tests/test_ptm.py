"""Sanity tests for Choi-to-PTM conversion."""
import numpy as np
import pytest
from physics.channels.ptm import choi_to_ptm, choi_to_ptm_features


def _identity_choi(d):
    """Choi of identity channel: |Phi+><Phi+| * d, where |Phi+> = sum_i |ii>."""
    phi = np.zeros((d * d,), dtype=complex)
    for i in range(d):
        phi[i * d + i] = 1.0
    return np.outer(phi, phi.conj())


def test_identity_channel_gives_identity_ptm_d2():
    R = choi_to_ptm(_identity_choi(2), d=2)
    assert np.allclose(R, np.eye(4), atol=1e-10), R


def test_identity_channel_gives_identity_ptm_d4():
    R = choi_to_ptm(_identity_choi(4), d=4)
    assert np.allclose(R, np.eye(16), atol=1e-10)


def test_depolarizing_channel_d2():
    # Depolarising channel with strength p:
    #   Lambda(rho) = (1-p) rho + p I/2
    # PTM: diag(1, 1-p, 1-p, 1-p)
    p = 0.3
    d = 2
    # Choi of depolarising:
    #   C = (1-p) |Phi+><Phi+| + p (I/d) (x) (I/d) * d^2
    #     = (1-p) |Phi+><Phi+| + p I_{d^2}
    Phi = _identity_choi(d)
    C = (1 - p) * Phi + p * np.eye(d * d) / d  # trace_out=I
    R = choi_to_ptm(C, d=d)
    expected = np.diag([1.0, 1 - p, 1 - p, 1 - p])
    assert np.allclose(R, expected, atol=1e-10), (R, expected)


def test_ptm_features_length():
    assert choi_to_ptm_features(_identity_choi(2), d=2).shape == (16,)
    assert choi_to_ptm_features(_identity_choi(4), d=4).shape == (256,)
