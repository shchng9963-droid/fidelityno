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
    quantum_shots: int     # total simulated projective measurements
    n_unique_paulis: int   # number of distinct Pauli settings measured
    strategy: str          # "iid" or "stratified"


def _sample_pauli_mean(
    expectation: np.ndarray,
    shots: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample means of +/-1 outcomes using their exact binomial law."""
    expectation = np.clip(np.asarray(expectation, dtype=np.float64), -1.0, 1.0)
    shots = np.asarray(shots, dtype=np.int64)
    if np.any(shots < 1):
        raise ValueError("each measured Pauli setting requires at least one shot")
    plus = rng.binomial(shots, 0.5 * (1.0 + expectation))
    return 2.0 * plus / shots - 1.0


def _allocate_stratified_shots(
    weights: np.ndarray,
    total_shots: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate a fixed shot budget across every nonzero-relevance Pauli."""
    support = np.flatnonzero(weights > 0)
    if total_shots < len(support):
        raise ValueError(
            f"stratified DFE needs at least {len(support)} shots, got {total_shots}"
        )
    shots = np.ones(len(support), dtype=np.int64)
    remaining = total_shots - len(support)
    if remaining:
        probs = np.sqrt(weights[support])
        probs /= probs.sum()
        expected = remaining * probs
        extra = np.floor(expected).astype(np.int64)
        shots += extra
        leftover = remaining - int(extra.sum())
        if leftover:
            order = np.argsort(-(expected - extra), kind="stable")
            shots[order[:leftover]] += 1
    return support, shots


def direct_fidelity_estimate(
    channels: list[Channel],
    target_unitary: Optional[np.ndarray] = None,
    *,
    num_paulis: int = 50,
    M_per_pauli: int = 200,
    noise: str = "finite",
    strategy: str = "iid",
    total_shots: Optional[int] = None,
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
    strategy:        "iid" samples Paulis with replacement; "stratified"
                     enumerates every nonzero-relevance Pauli.
    total_shots:     fixed budget for stratified DFE. Defaults to
                     ``num_paulis * M_per_pauli``.
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
    if noise not in {"exact", "finite"}:
        raise ValueError(noise)
    S = int(num_paulis)
    if S < 1 or M_per_pauli < 1:
        raise ValueError("num_paulis and M_per_pauli must be positive")

    if strategy == "iid":
        idx = rng.choice(len(chi_V), size=S, replace=True, p=weights)
        observed = chi_L[idx]
        if noise == "finite":
            observed = _sample_pauli_mean(observed, np.full(S, M_per_pauli), rng)
        X = observed / chi_V[idx]
        F_hat = float(X.mean())
        sigma_proto = float(X.std(ddof=1) / np.sqrt(S)) if S > 1 else float("nan")
        quantum_shots = S * M_per_pauli if noise == "finite" else 0
        n_unique = int(np.unique(idx).size)
    elif strategy == "stratified":
        budget = S * M_per_pauli if total_shots is None else int(total_shots)
        idx, shots = _allocate_stratified_shots(weights, budget)
        observed = chi_L[idx]
        if noise == "finite":
            observed = _sample_pauli_mean(observed, shots, rng)
        F_hat = float(np.sum(observed * chi_V[idx]) / d ** 2)
        variances = np.maximum(1.0 - np.square(chi_L[idx]), 0.0) / shots
        sigma_proto = float(np.sqrt(np.sum(np.square(chi_V[idx]) * variances)) / d ** 2)
        quantum_shots = budget if noise == "finite" else 0
        n_unique = len(idx)
    else:
        raise ValueError(f"unknown DFE strategy: {strategy}")

    return DFEResult(
        F_hat=F_hat,
        F_exact=F_exact,
        abs_err=abs(F_hat - F_exact),
        sigma_protocol=sigma_proto,
        n_paulis=S,
        M_per_pauli=M_per_pauli,
        quantum_shots=quantum_shots,
        n_unique_paulis=n_unique,
        strategy=strategy,
    )


__all__ = [
    "DFEResult",
    "all_pauli_labels",
    "chi_channel",
    "chi_unitary",
    "direct_fidelity_estimate",
    "_allocate_stratified_shots",
]
