"""Utilities for information-limited and measurement-conditioned estimators."""
from __future__ import annotations

import numpy as np

from physics.baselines.dfe import PAULIS, _allocate_stratified_shots


def batch_pauli_expectations_from_choi(choi: np.ndarray) -> np.ndarray:
    """Return single-qubit ``chi_L(P)`` for I, X, Y, Z from batched Choi data."""
    choi = np.asarray(choi, dtype=np.complex128)
    if choi.ndim == 2:
        choi = choi[None, ...]
    if choi.ndim != 3 or choi.shape[1:] != (4, 4):
        raise ValueError("choi must have shape (N, 4, 4) or (4, 4)")
    blocks = choi.reshape(len(choi), 2, 2, 2, 2)  # N, input-i, out-a, input-j, out-b
    values = np.empty((len(choi), 4), dtype=np.float64)
    for index, pauli in enumerate(PAULIS):
        applied = np.einsum("ij,niajb->nab", pauli, blocks, optimize=True)
        values[:, index] = np.einsum("ab,nba->n", pauli, applied, optimize=True).real / 2.0
    return np.clip(values, -1.0, 1.0)


def sample_identity_dfe(
    expectations: np.ndarray,
    total_shots: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised stratified DFE for a single-qubit identity target.

    Returns the fidelity estimates and their oracle conditional standard
    deviations.  The latter are for diagnostics only and are not used as
    model inputs by the hybrid estimator.
    """
    expectations = np.asarray(expectations, dtype=np.float64)
    if expectations.ndim != 2 or expectations.shape[1] != 4:
        raise ValueError("expectations must have shape (N, 4)")
    support, shots = _allocate_stratified_shots(np.full(4, 0.25), int(total_shots))
    if not np.array_equal(support, np.arange(4)):
        raise RuntimeError("identity-target DFE must use all four Paulis")
    probs = np.clip(0.5 * (1.0 + expectations), 0.0, 1.0)
    plus = rng.binomial(shots[None, :], probs)
    observed = 2.0 * plus / shots[None, :] - 1.0
    estimate = observed.mean(axis=1)
    variance = np.sum((1.0 - np.square(expectations)) / shots[None, :], axis=1) / 16.0
    return estimate, np.sqrt(np.maximum(variance, 0.0))


def fit_convex_fusion(
    prior: np.ndarray,
    measurement: np.ndarray,
    target: np.ndarray,
) -> float:
    """Fit ``(1-w) prior + w measurement`` by constrained least squares."""
    prior = np.asarray(prior, dtype=np.float64)
    measurement = np.asarray(measurement, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prior.shape != measurement.shape or prior.shape != target.shape:
        raise ValueError("prior, measurement, and target must share a shape")
    direction = measurement - prior
    denom = float(np.dot(direction, direction))
    if denom <= 1e-15:
        return 0.0
    weight = float(np.dot(direction, target - prior) / denom)
    return float(np.clip(weight, 0.0, 1.0))


def apply_convex_fusion(prior: np.ndarray, measurement: np.ndarray, weight: float) -> np.ndarray:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must lie in [0, 1]")
    return (1.0 - weight) * np.asarray(prior) + weight * np.asarray(measurement)


def ambiguity_statistics(values: np.ndarray) -> dict[str, np.ndarray]:
    """Pointwise minimax and uniform-grid Bayes quantities for counterfactuals."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("values must have shape (N, K) with K >= 2")
    lower = values.min(axis=1)
    upper = values.max(axis=1)
    median = np.median(values, axis=1)
    midpoint = 0.5 * (lower + upper)
    return {
        "lower": lower,
        "upper": upper,
        "diameter": upper - lower,
        "minimax_abs_lower_bound": 0.5 * (upper - lower),
        "conditional_median": median,
        "uniform_grid_bayes_mae": np.abs(values - median[:, None]).mean(axis=1),
        "minimax_midpoint": midpoint,
    }


__all__ = [
    "ambiguity_statistics",
    "apply_convex_fusion",
    "batch_pauli_expectations_from_choi",
    "fit_convex_fusion",
    "sample_identity_dfe",
]
