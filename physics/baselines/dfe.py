"""Direct Fidelity Estimation (DFE) baseline.

Implements the Flammia-Liu (PRL 106, 230501, 2011) protocol for estimating
the entanglement fidelity F_e(Lambda, V) between a target unitary V and a
noisy implementation Lambda from a small number of Pauli measurements.

We support two "noise" levels for the simulated hardware reference:

  noise="exact"   -- compute chi_Lambda(P) exactly from the composed Choi.
                     This is what you would get with an INFINITE number of
                     shots per Pauli; useful for comparing the DFE
                     *protocol variance* alone.
  noise="finite"  -- additionally inject the per-Pauli sampling noise from
                     a finite number of repetitions M of the projective
                     measurement on hardware (default M=200).  This is the
                     real-experiment estimator.

For an n-qubit register acting on dimension d=2**n the protocol is:

  1. Build the characteristic function chi_V(P) = Tr[P V P V^dagger] / d
     for the IDEAL target.  For V = I (the case for noise channels), this
     gives chi_V(P) = 1 for all P, so the relevance distribution is
     uniform; for a general target unitary V, importance sampling sharply
     reduces variance.
  2. Compute chi_V(P)^2 / d^2 to get the importance distribution Pr(P).
  3. Sample S Paulis from Pr(P).  For each, compute chi_Lambda(P) =
     Tr[P Lambda(P)] / d.
  4. Average X_i = chi_Lambda(P_i) / chi_V(P_i) to estimate F_e:
       Fhat = (1/S) sum_i X_i.

Cited:  Flammia & Liu, PRL 106, 230501 (2011); da Silva, Landon-Cardinal,
Poulin, PRL 107, 210404 (2011).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from physics.channels.base import Channel
from physics.composition import compose_channels


# Single-qubit Paulis (column-major order I, X, Y, Z)
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = (_I, _X, _Y, _Z)


def _kron_paulis(label: tuple[int, ...]) -> np.ndarray:
    """Kronecker product of single-qubit Paulis indexed by 0..3 = I,X,Y,Z."""
    out = PAULIS[label[0]]
    for k in label[1:]:
        out = np.kron(out, PAULIS[k])
    return out


def all_pauli_labels(num_qubits: int) -> list[tuple[int, ...]]:
    return list(itertools.product(range(4), repeat=num_qubits))


def chi_unitary(U: np.ndarray, num_qubits: int) -> np.ndarray:
    """chi_V(P) = Tr[P V P V^dagger] / d for all Paulis P.

    Returns shape (4^n,), in lexicographic Pauli order.
    """
    d = 1 << num_qubits
    labels = all_pauli_labels(num_qubits)
    out = np.empty(len(labels), dtype=float)
    for i, lab in enumerate(labels):
        P = _kron_paulis(lab)
        # Tr[P V P V^dagger] = Tr[(V^dagger P V) P]
        VPVd = U.conj().T @ P @ U
        out[i] = float(np.real(np.trace(P @ VPVd)) / d)
    return out


def chi_channel(channel: Channel, num_qubits: int) -> np.ndarray:
    """chi_Lambda(P) = Tr[P Lambda(P)] / d for all Paulis P."""
    d = 1 << num_qubits
    assert channel.dim == d, (channel.dim, d)
    labels = all_pauli_labels(num_qubits)
    out = np.empty(len(labels), dtype=float)
    for i, lab in enumerate(labels):
        P = _kron_paulis(lab)
        LP = channel.apply(P)
        out[i] = float(np.real(np.trace(P @ LP)) / d)
    return out


@dataclass
class DFEResult:
    F_hat: float           # estimated entanglement fidelity
    F_exact: float         # exact F_e from composed Choi
    abs_err: float         # |F_hat - F_exact|
    sigma_protocol: float  # sample stddev across S Paulis (importance-sampled estimator)
    n_paulis: int          # S
    M_per_pauli: int       # measurement repetitions per Pauli (>=1)


def direct_fidelity_estimate(
    channels: list[Channel],
    target_unitary: Optional[np.ndarray] = None,
    *,
    num_paulis: int = 50,
    M_per_pauli: int = 200,
    noise: str = "finite",
    rng: Optional[np.random.Generator] = None,
) -> DFEResult:
    """Run DFE on the *composed* channel.

    Parameters
    ----------
    channels:        ordered list of `Channel` objects in the cascade.
    target_unitary:  ideal unitary V on the same dimension. If None, V = I
                     (noise-channel case).
    num_paulis:      S, number of importance-sampled Paulis.
    M_per_pauli:     M, projective-measurement repetitions per Pauli, used
                     to inject finite-shot variance when noise="finite".
    noise:           "exact" or "finite".
    rng:             numpy Generator (defaults to default_rng()).

    Returns
    -------
    DFEResult
    """
    if rng is None:
        rng = np.random.default_rng()
    composed = compose_channels(channels)
    d = composed.dim
    num_qubits = int(round(np.log2(d)))
    assert 1 << num_qubits == d, "DFE requires power-of-2 dimensions"

    if target_unitary is None:
        target_unitary = np.eye(d, dtype=complex)

    chi_V = chi_unitary(target_unitary, num_qubits)        # (4^n,)
    chi_L = chi_channel(composed, num_qubits)              # (4^n,)
    F_exact = float(np.dot(chi_L, chi_V) / d ** 2)

    # Importance sampling: Pr(P) = chi_V(P)^2 / d^2
    weights = chi_V ** 2 / d ** 2
    weights = weights / weights.sum()  # numerical safety
    S = num_paulis
    idx = rng.choice(len(chi_V), size=S, replace=True, p=weights)

    # X_i = chi_Lambda(P_i) / chi_V(P_i)
    chi_L_sample = chi_L[idx]
    chi_V_sample = chi_V[idx]
    X = chi_L_sample / chi_V_sample

    if noise == "finite":
        # The eigenvalues of P are +-1, so chi_Lambda(P) is the expectation
        # of a +-1 valued random variable.  After M measurements its
        # variance is at most (1 - chi_L^2)/M, which we bound by 1/M.
        sigma = np.sqrt(np.maximum(1.0 - chi_L_sample ** 2, 0.0) / M_per_pauli)
        chi_L_noisy = chi_L_sample + rng.normal(0.0, sigma)
        X = chi_L_noisy / chi_V_sample
    elif noise != "exact":
        raise ValueError(noise)

    F_hat = float(X.mean())
    sigma_proto = float(X.std(ddof=1) / np.sqrt(S)) if S > 1 else float("nan")
    return DFEResult(
        F_hat=F_hat,
        F_exact=F_exact,
        abs_err=abs(F_hat - F_exact),
        sigma_protocol=sigma_proto,
        n_paulis=S,
        M_per_pauli=M_per_pauli,
    )


__all__ = [
    "DFEResult",
    "all_pauli_labels",
    "chi_channel",
    "chi_unitary",
    "direct_fidelity_estimate",
]
