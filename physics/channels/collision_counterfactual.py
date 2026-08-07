"""Counterfactual collision sequences with fixed microscopic parameters."""
from __future__ import annotations

from typing import Optional

import numpy as np

from physics.channels.base import Channel
from physics.channels.collision_nonmarkov import (
    PLUS,
    CollisionSample,
    _collision_unitary,
    _marginal_choi_from_unitary,
    _system_choi_from_joint_propagation,
)
from physics.fidelity import entanglement_fidelity


def _validated_params(params: np.ndarray) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64)
    if params.ndim != 2 or params.shape[1] != 3 or len(params) < 1:
        raise ValueError("params must have shape (L, 3) with L >= 1")
    if not np.isfinite(params).all():
        raise ValueError("params must be finite")
    return params


def collision_fidelity_grid_from_params(
    params: np.ndarray,
    eta_values: np.ndarray,
    *,
    rho_B_ref: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Evaluate fidelities over eta while reusing the same collision unitaries."""
    params = _validated_params(params)
    eta_values = np.asarray(eta_values, dtype=np.float64)
    if eta_values.ndim != 1 or len(eta_values) < 1:
        raise ValueError("eta_values must be a non-empty vector")
    if np.any((eta_values < 0.0) | (eta_values > 1.0)):
        raise ValueError("all eta values must lie in [0, 1]")
    if rho_B_ref is None:
        rho_B_ref = PLUS.copy()
    unitaries = [_collision_unitary(J, omega, tau) for J, omega, tau in params]
    fidelities = np.empty(len(eta_values), dtype=np.float64)
    for index, eta in enumerate(eta_values):
        true_choi = _system_choi_from_joint_propagation(
            unitaries,
            eta=float(eta),
            rho_B_init=rho_B_ref,
        )
        fidelities[index] = entanglement_fidelity(
            Channel(name="true_overall", dim=2, choi=true_choi)
        )
    return fidelities


def collision_sequence_from_params(
    params: np.ndarray,
    *,
    eta: float,
    rho_B_ref: Optional[np.ndarray] = None,
) -> CollisionSample:
    """Replay a collision sequence while varying only bath retention ``eta``."""
    params = _validated_params(params)
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must lie in [0, 1]")
    if rho_B_ref is None:
        rho_B_ref = PLUS.copy()
    else:
        rho_B_ref = np.asarray(rho_B_ref, dtype=np.complex128)
        if rho_B_ref.shape != (2, 2):
            raise ValueError("rho_B_ref must have shape (2, 2)")

    unitaries = [_collision_unitary(J, omega, tau) for J, omega, tau in params]
    marginals = []
    for index, unitary in enumerate(unitaries):
        choi = _marginal_choi_from_unitary(unitary, rho_B_ref)
        marginals.append(
            Channel(
                name=f"collision_{index}",
                dim=2,
                choi=choi,
                params=params[index].copy(),
            )
        )

    true_choi = _system_choi_from_joint_propagation(
        unitaries,
        eta=float(eta),
        rho_B_init=rho_B_ref,
    )
    true_channel = Channel(name="true_overall", dim=2, choi=true_choi)
    return CollisionSample(
        marginals=marginals,
        true_F_e=float(entanglement_fidelity(true_channel)),
        true_choi=true_choi,
        eta=float(eta),
        params=params.copy(),
    )


__all__ = ["collision_fidelity_grid_from_params", "collision_sequence_from_params"]
