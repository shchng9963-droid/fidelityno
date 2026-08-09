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


def sample_identity_dfe_readout(
    expectations: np.ndarray,
    total_shots: int,
    readout_error: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample DFE with symmetric outcome flips and calibrated mitigation.

    Each ideal Pauli outcome is independently sign-flipped with probability
    ``readout_error``. The raw estimate uses the observed outcomes directly.
    The mitigated estimate divides each observed Pauli expectation by the
    calibrated attenuation ``1 - 2 * readout_error`` before averaging.
    Both estimates are returned from the same counts for paired comparisons.
    """
    expectations = np.asarray(expectations, dtype=np.float64)
    if expectations.ndim != 2 or expectations.shape[1] != 4:
        raise ValueError("expectations must have shape (N, 4)")
    if not 0.0 <= readout_error < 0.5:
        raise ValueError("readout_error must lie in [0, 0.5)")
    support, shots = _allocate_stratified_shots(np.full(4, 0.25), int(total_shots))
    if not np.array_equal(support, np.arange(4)):
        raise RuntimeError("identity-target DFE must use all four Paulis")

    attenuation = 1.0 - 2.0 * readout_error
    observed_expectations = attenuation * expectations
    probs = np.clip(0.5 * (1.0 + observed_expectations), 0.0, 1.0)
    plus = rng.binomial(shots[None, :], probs)
    observed = 2.0 * plus / shots[None, :] - 1.0
    raw = observed.mean(axis=1)
    # Linear inversion is intentionally left unclipped at the observable
    # level. Finite-shot mitigated estimates can lie outside the physical
    # interval; clipping is applied only to the final fidelity prediction.
    mitigated_observed = observed / attenuation
    mitigated = mitigated_observed.mean(axis=1)

    raw_variance = (
        np.sum(
            (1.0 - np.square(observed_expectations)) / shots[None, :], axis=1
        )
        / 16.0
    )
    mitigated_variance = raw_variance / np.square(attenuation)
    return (
        raw,
        mitigated,
        np.sqrt(np.maximum(raw_variance, 0.0)),
        np.sqrt(np.maximum(mitigated_variance, 0.0)),
    )


def sample_identity_dfe_pilot(
    expectations: np.ndarray,
    total_shots: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a reusable stratified pilot and return plus counts and estimates."""
    expectations = np.asarray(expectations, dtype=np.float64)
    if expectations.ndim != 2 or expectations.shape[1] != 4:
        raise ValueError("expectations must have shape (N, 4)")
    if total_shots < 4 or total_shots % 4:
        raise ValueError("pilot total_shots must be a positive multiple of four")
    shots_per_setting = total_shots // 4
    probs = np.clip(0.5 * (1.0 + expectations), 0.0, 1.0)
    plus = rng.binomial(shots_per_setting, probs)
    observed = 2.0 * plus / shots_per_setting - 1.0
    return plus.astype(np.int64), observed.mean(axis=1)


def complete_identity_dfe(
    expectations: np.ndarray,
    pilot_plus: np.ndarray,
    pilot_shots: int,
    final_total_shots: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Complete a nested stratified DFE estimate at query-specific budgets."""
    expectations = np.asarray(expectations, dtype=np.float64)
    pilot_plus = np.asarray(pilot_plus, dtype=np.int64)
    final_total_shots = np.asarray(final_total_shots, dtype=np.int64)
    if expectations.ndim != 2 or expectations.shape[1] != 4:
        raise ValueError("expectations must have shape (N, 4)")
    if pilot_plus.shape != expectations.shape:
        raise ValueError("pilot_plus must match expectations")
    if final_total_shots.shape != (len(expectations),):
        raise ValueError("final_total_shots must have shape (N,)")
    if pilot_shots < 4 or pilot_shots % 4:
        raise ValueError("pilot_shots must be a positive multiple of four")
    if np.any(final_total_shots < pilot_shots) or np.any(final_total_shots % 4):
        raise ValueError("final budgets must be multiples of four and include the pilot")

    pilot_per_setting = pilot_shots // 4
    final_per_setting = final_total_shots // 4
    extra_per_setting = final_per_setting - pilot_per_setting
    probs = np.clip(0.5 * (1.0 + expectations), 0.0, 1.0)
    extra_plus = rng.binomial(extra_per_setting[:, None], probs)
    total_plus = pilot_plus + extra_plus
    observed = 2.0 * total_plus / final_per_setting[:, None] - 1.0
    estimate = observed.mean(axis=1)
    variance = (
        np.sum(
            (1.0 - np.square(expectations)) / final_per_setting[:, None], axis=1
        )
        / 16.0
    )
    return estimate, np.sqrt(np.maximum(variance, 0.0))


def allocate_two_level_budget(
    scores: np.ndarray,
    low_shots: int,
    high_shots: int,
    high_fraction: float = 0.5,
) -> np.ndarray:
    """Assign the high budget to the largest scores with deterministic ties."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite one-dimensional array")
    if low_shots < 4 or low_shots % 4 or high_shots <= low_shots or high_shots % 4:
        raise ValueError("shot levels must be distinct positive multiples of four")
    if not 0.0 < high_fraction < 1.0:
        raise ValueError("high_fraction must lie in (0, 1)")
    n_high = int(round(high_fraction * len(scores)))
    n_high = min(max(n_high, 1), len(scores) - 1)
    order = np.lexsort((np.arange(len(scores)), scores))
    budgets = np.full(len(scores), low_shots, dtype=np.int64)
    budgets[order[-n_high:]] = high_shots
    return budgets


def fit_budgeted_convex_fusion(
    prior: np.ndarray,
    measurement: np.ndarray,
    target: np.ndarray,
    budgets: np.ndarray,
) -> dict[int, float]:
    """Fit one constrained fusion weight for each realised shot budget."""
    prior = np.asarray(prior, dtype=np.float64)
    measurement = np.asarray(measurement, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    budgets = np.asarray(budgets, dtype=np.int64)
    if not (prior.shape == measurement.shape == target.shape == budgets.shape):
        raise ValueError("prior, measurement, target, and budgets must share a shape")
    return {
        int(budget): fit_convex_fusion(
            prior[budgets == budget],
            measurement[budgets == budget],
            target[budgets == budget],
        )
        for budget in np.unique(budgets)
    }


def apply_budgeted_convex_fusion(
    prior: np.ndarray,
    measurement: np.ndarray,
    budgets: np.ndarray,
    weights: dict[int, float],
) -> np.ndarray:
    """Apply shot-specific convex weights fitted on a calibration set."""
    prior = np.asarray(prior, dtype=np.float64)
    measurement = np.asarray(measurement, dtype=np.float64)
    budgets = np.asarray(budgets, dtype=np.int64)
    if not (prior.shape == measurement.shape == budgets.shape):
        raise ValueError("prior, measurement, and budgets must share a shape")
    result = np.empty_like(prior)
    for budget in np.unique(budgets):
        if int(budget) not in weights:
            raise ValueError(f"missing fusion weight for budget {budget}")
        mask = budgets == budget
        result[mask] = apply_convex_fusion(
            prior[mask], measurement[mask], weights[int(budget)]
        )
    return result


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
    "allocate_two_level_budget",
    "ambiguity_statistics",
    "apply_budgeted_convex_fusion",
    "apply_convex_fusion",
    "batch_pauli_expectations_from_choi",
    "complete_identity_dfe",
    "fit_budgeted_convex_fusion",
    "fit_convex_fusion",
    "sample_identity_dfe",
    "sample_identity_dfe_readout",
    "sample_identity_dfe_pilot",
]
