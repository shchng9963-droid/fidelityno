"""Normalised Choi trace-distance bounds for the specified channel only."""
import numpy as np


def choi_trace_distance_lower_bound(choi, target_unitary=None):
    """Return max(0,1-||J/d-J_target/d||_1/2) for a pure unitary target.

    The bound concerns the given Choi matrix. Composing reset-bath marginals
    does not certify the fidelity of a different retained-bath process.
    """
    choi = np.asarray(choi, dtype=complex)
    d = int(round(np.sqrt(choi.shape[0])))
    if choi.shape != (d*d, d*d):
        raise ValueError("Choi matrix must have shape (d^2,d^2)")
    if not np.allclose(choi, choi.conj().T, atol=1e-6):
        raise ValueError("Choi matrix must be Hermitian")
    if not np.isclose(np.trace(choi), d, atol=1e-6):
        raise ValueError("unnormalised Choi matrix must have trace d")
    if np.linalg.eigvalsh(choi).min() < -1e-6:
        raise ValueError("Choi matrix must be positive semidefinite")
    u = np.eye(d) if target_unitary is None else np.asarray(target_unitary)
    if u.shape != (d,d) or not np.allclose(u.conj().T @ u, np.eye(d), atol=1e-7):
        raise ValueError("reference must be unitary")
    # Input-first Choi convention: vec(U) in column-major order.
    v = u.reshape(-1, order="F")
    delta = (choi - np.outer(v, v.conj())) / d
    trace_distance = np.abs(np.linalg.eigvalsh((delta + delta.conj().T)/2)).sum()/2
    return float(np.clip(1-trace_distance, 0, 1))
