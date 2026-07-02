"""Non-Markovian collision-model channel family.

Setup (Ciccarello et al., Phys. Rep. 954, 2022, sect. 3 + 5):

  System S (qubit), bath B (single qubit).  At each collision t a system-
  bath unitary

      U_t = exp[-i tau_t * ( J_t * Z_S Z_B + omega_t * Z_B / 2 )]

  is applied to S + B.  Between collisions the bath has retention
  probability eta in [0, 1]:

      rho_B(t -> t+1) = eta * rho_B(after collision t)
                       + (1 - eta) * |+><+|.

  - eta = 0 reproduces the Markovian limit: each Λ_t is a fixed CPTP
    map and the per-step marginal channels compose multiplicatively.
  - eta = 1 makes the dynamics fully non-Markovian: the bath retains
    correlation with the system's history.  The per-step "marginal
    channel" the surrogate sees is the channel that *would* have acted
    if the bath had been reset to |+>; the true sequence-level
    fidelity differs from any product/composition computed from those
    marginals.

The function `collision_sequence(...)` returns

    (marginal_channels, true_F_e)

where marginal_channels are CPTP `Channel` objects (the surrogate's
input) and true_F_e is the entanglement fidelity of the *true* system
marginal channel after full joint propagation.

For the surrogate to predict true_F_e, it must learn the bath-retention
correction; the analytic product bound provably cannot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from physics.channels.base import Channel
from physics.composition import compose_channels
from physics.fidelity import entanglement_fidelity


# Single-qubit Paulis
I2 = np.eye(2, dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PLUS = 0.5 * (I2 + X)  # |+><+| = (I + X)/2


def _collision_unitary(J: float, omega: float, tau: float) -> np.ndarray:
    """U_t on (S,B) = exp[-i tau (J Z⊗Z + omega/2 (I⊗Z))]."""
    H = J * np.kron(Z, Z) + 0.5 * omega * np.kron(I2, Z)
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvecs @ np.diag(np.exp(-1j * tau * eigvals)) @ eigvecs.conj().T


def _marginal_choi_from_unitary(U: np.ndarray, rho_B_ref: np.ndarray) -> np.ndarray:
    """Build the Choi matrix of  rho -> Tr_B[ U (rho ⊗ rho_B_ref) U† ]."""
    # Apply to basis operators E_ij = |i><j| and stack.
    # Choi C = sum_{ij} |i><j| ⊗ Λ(|i><j|) is the standard convention used
    # in this repo (see physics/channels/base.choi_to_superop).
    d_S = 2
    C = np.zeros((d_S * d_S, d_S * d_S), dtype=np.complex128)
    for i in range(d_S):
        for j in range(d_S):
            E = np.zeros((d_S, d_S), dtype=np.complex128)
            E[i, j] = 1.0
            joint_in = np.kron(E, rho_B_ref)
            joint_out = U @ joint_in @ U.conj().T
            # partial trace over bath -> 2x2 matrix
            sys_out = np.zeros((d_S, d_S), dtype=np.complex128)
            for b in range(d_S):
                sys_out += joint_out[b::d_S, b::d_S]
            # write into block (i,j)
            C[i * d_S:(i + 1) * d_S, j * d_S:(j + 1) * d_S] = sys_out
    return C


def _partial_trace_bath(joint: np.ndarray, d_S: int) -> np.ndarray:
    """Trace out the bath subsystem (rightmost factor).  joint is shape
    (d_S * d_B, d_S * d_B) with d_B = d_S = 2 here; returns (d_S, d_S).
    """
    d_B = joint.shape[0] // d_S
    out = np.zeros((d_S, d_S), dtype=joint.dtype)
    for b in range(d_B):
        out += joint[b::d_B, b::d_B]
    return out


def _partial_trace_system(joint: np.ndarray, d_S: int) -> np.ndarray:
    """Trace out the system subsystem (leftmost factor)."""
    d_B = joint.shape[0] // d_S
    out = np.zeros((d_B, d_B), dtype=joint.dtype)
    for s in range(d_S):
        out += joint[s * d_B:(s + 1) * d_B, s * d_B:(s + 1) * d_B]
    return out


def _system_choi_from_joint_propagation(
    Us: list[np.ndarray],
    eta: float,
    rho_B_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """True system Choi after the joint propagation under the collision sequence.

    Bath-refresh model (Ciccarello et al. 2022, eq. 5.2):
       After each collision t, with probability (1 - eta) the bath is
       reset to ``rho_B_ref``; with probability eta the joint S+B state
       is retained intact:

           joint(t+1) = eta * joint(t+0.5)
                        + (1 - eta) * (Tr_B[joint(t+0.5)] ⊗ rho_B_ref)

    This mixture is linear in the system input, so the resulting overall
    map ρ_S(0) -> ρ_S(n) is CPTP.  At eta=0 the model is Markovian (bath
    reset to rho_B_ref every step, matching the surrogate's marginal
    view); at eta=1 the joint state is kept forever, encoding genuine
    non-Markovianity.
    """
    if rho_B_init is None:
        rho_B_init = PLUS.copy()
    d_S = 2
    d_B = 2

    C = np.zeros((d_S * d_S, d_S * d_S), dtype=np.complex128)
    for i in range(d_S):
        for j in range(d_S):
            E = np.zeros((d_S, d_S), dtype=np.complex128)
            E[i, j] = 1.0
            joint = np.kron(E, rho_B_init)
            for U in Us:
                joint = U @ joint @ U.conj().T
                if eta < 1.0:
                    sys_marginal = _partial_trace_bath(joint, d_S)
                    joint = eta * joint + (1.0 - eta) * np.kron(sys_marginal, rho_B_init)
            final_sys = _partial_trace_bath(joint, d_S)
            C[i * d_S:(i + 1) * d_S, j * d_S:(j + 1) * d_S] = final_sys
    return C


@dataclass
class CollisionSample:
    marginals: list[Channel]
    true_F_e: float
    true_choi: np.ndarray   # shape (d^2, d^2) Choi matrix of the true non-Markovian channel
    eta: float
    params: np.ndarray  # (n, 3) [J, omega, tau] for each step


def collision_sequence(
    num_collisions: int,
    *,
    J_range: tuple[float, float] = (0.05, 0.20),
    omega_range: tuple[float, float] = (0.10, 0.40),
    tau_range: tuple[float, float] = (0.30, 1.00),
    eta: float = 0.6,
    rho_B_ref: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> CollisionSample:
    """Build one non-Markovian collision-model sample.

    Parameters
    ----------
    num_collisions: sequence length n.
    J_range, omega_range, tau_range: parameter ranges for collisions.
    eta: bath-retention probability.  Set to 0 for Markovian sanity check.
    rho_B_ref: reference bath state for the *surrogate's* per-step
               marginal channel.  Default |+><+|.
    """
    if rng is None:
        rng = np.random.default_rng()
    if rho_B_ref is None:
        rho_B_ref = PLUS.copy()

    Js = rng.uniform(*J_range, size=num_collisions)
    omegas = rng.uniform(*omega_range, size=num_collisions)
    taus = rng.uniform(*tau_range, size=num_collisions)
    params = np.stack([Js, omegas, taus], axis=1)

    # Per-collision unitary
    Us = [_collision_unitary(J, om, t) for J, om, t in zip(Js, omegas, taus)]

    # Surrogate's view: each collision marginal computed assuming the
    # bath is in rho_B_ref every time.
    marginals: list[Channel] = []
    for i, U in enumerate(Us):
        choi = _marginal_choi_from_unitary(U, rho_B_ref)
        marginals.append(Channel(name=f"collision_{i}", dim=2, choi=choi,
                                 params=np.array([Js[i], omegas[i], taus[i]])))

    # Ground-truth: full joint propagation with bath retention eta
    true_choi = _system_choi_from_joint_propagation(Us, eta=eta,
                                                    rho_B_init=rho_B_ref)
    true_channel = Channel(name="true_overall", dim=2, choi=true_choi)
    true_F_e = float(entanglement_fidelity(true_channel))

    return CollisionSample(
        marginals=marginals,
        true_F_e=true_F_e,
        true_choi=true_choi,
        eta=eta,
        params=params,
    )


__all__ = ["CollisionSample", "collision_sequence"]
