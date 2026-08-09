"""Exchange-coupled collision model with a retained bath qubit.

This family is a representation audit for the information-limited setting.
It differs from the dephasing collision family in
``collision_nonmarkov.py`` because the system and bath exchange excitations.
The local bath precession does not commute with the exchange term.  The
retention coefficient is hidden from the list of reset-bath marginal
channels, so the same observable input can correspond to different overall
channel fidelities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from physics.channels.base import Channel
from physics.channels.collision_nonmarkov import (
    _marginal_choi_from_unitary,
    _system_choi_from_joint_propagation,
)
from physics.fidelity import entanglement_fidelity


I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
DEFAULT_BATH = np.diag([0.8, 0.2]).astype(np.complex128)


def _exchange_hamiltonian(
    coupling: float,
    detuning: float,
    bath_frequency: float,
) -> np.ndarray:
    """Return an anisotropic exchange Hamiltonian on system and bath."""
    exchange = 0.5 * coupling * (np.kron(X, X) + np.kron(Y, Y))
    dispersive = 0.5 * detuning * np.kron(Z, Z)
    bath_precession = 0.5 * bath_frequency * np.kron(I2, Z)
    return exchange + dispersive + bath_precession


def _exchange_unitary(
    coupling: float,
    detuning: float,
    bath_frequency: float,
    duration: float,
) -> np.ndarray:
    hamiltonian = _exchange_hamiltonian(coupling, detuning, bath_frequency)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    return eigenvectors @ np.diag(np.exp(-1j * duration * eigenvalues)) @ eigenvectors.conj().T


@dataclass
class ExchangeMemorySample:
    marginals: list[Channel]
    true_F_e: float
    true_choi: np.ndarray
    eta: float
    params: np.ndarray  # (n, 4): coupling, detuning, bath frequency, duration


def exchange_sequence_from_params(
    params: np.ndarray,
    *,
    eta: float,
    rho_B_ref: Optional[np.ndarray] = None,
) -> ExchangeMemorySample:
    """Replay one exchange-coupled sequence at a chosen retention value."""
    params = np.asarray(params, dtype=np.float64)
    if params.ndim != 2 or params.shape[1] != 4 or len(params) == 0:
        raise ValueError("params must have shape (n, 4) with n > 0")
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must lie in [0, 1]")
    if rho_B_ref is None:
        rho_B_ref = DEFAULT_BATH.copy()
    rho_B_ref = np.asarray(rho_B_ref, dtype=np.complex128)
    if rho_B_ref.shape != (2, 2):
        raise ValueError("rho_B_ref must have shape (2, 2)")

    unitaries = [_exchange_unitary(*row) for row in params]
    marginals = []
    for index, (unitary, row) in enumerate(zip(unitaries, params)):
        choi = _marginal_choi_from_unitary(unitary, rho_B_ref)
        marginals.append(
            Channel(
                name=f"exchange_collision_{index}",
                dim=2,
                choi=choi,
                params=row.copy(),
            )
        )

    true_choi = _system_choi_from_joint_propagation(
        unitaries,
        eta=eta,
        rho_B_init=rho_B_ref,
    )
    true_channel = Channel(name="true_exchange_overall", dim=2, choi=true_choi)
    return ExchangeMemorySample(
        marginals=marginals,
        true_F_e=float(entanglement_fidelity(true_channel)),
        true_choi=true_choi,
        eta=float(eta),
        params=params.copy(),
    )


def exchange_collision_sequence(
    num_collisions: int,
    *,
    coupling_range: tuple[float, float] = (0.08, 0.25),
    detuning_range: tuple[float, float] = (0.02, 0.12),
    bath_frequency_range: tuple[float, float] = (0.08, 0.30),
    duration_range: tuple[float, float] = (0.25, 0.90),
    eta: float = 0.6,
    rho_B_ref: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> ExchangeMemorySample:
    """Sample one sequence and return reset-bath marginals plus true output."""
    if num_collisions < 1:
        raise ValueError("num_collisions must be positive")
    if rng is None:
        rng = np.random.default_rng()
    params = np.column_stack(
        [
            rng.uniform(*coupling_range, size=num_collisions),
            rng.uniform(*detuning_range, size=num_collisions),
            rng.uniform(*bath_frequency_range, size=num_collisions),
            rng.uniform(*duration_range, size=num_collisions),
        ]
    )
    return exchange_sequence_from_params(params, eta=eta, rho_B_ref=rho_B_ref)


def exchange_fidelity_grid_from_params(
    params: np.ndarray,
    eta_grid: np.ndarray,
    *,
    rho_B_ref: Optional[np.ndarray] = None,
) -> np.ndarray:
    eta_grid = np.asarray(eta_grid, dtype=np.float64)
    if eta_grid.ndim != 1 or len(eta_grid) < 2:
        raise ValueError("eta_grid must be one-dimensional with at least two values")
    return np.array(
        [
            exchange_sequence_from_params(
                params,
                eta=float(eta),
                rho_B_ref=rho_B_ref,
            ).true_F_e
            for eta in eta_grid
        ],
        dtype=np.float64,
    )


__all__ = [
    "DEFAULT_BATH",
    "ExchangeMemorySample",
    "exchange_collision_sequence",
    "exchange_fidelity_grid_from_params",
    "exchange_sequence_from_params",
]
