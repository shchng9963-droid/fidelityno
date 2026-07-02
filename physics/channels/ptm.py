"""Pauli-transfer-matrix (PTM) encoder utility.

For a CPTP map Lambda on a single-qubit (d=2) system, the PTM is the 4x4 real matrix
   R_{ij} = (1/d) Tr[P_i Lambda(P_j)],   i,j in {I, X, Y, Z}
where P_i are Paulis (unnormalised). PTM is real and entries are bounded in [-1,1] by CPTP.
We flatten the PTM as 16 real numbers per channel.

For d=4 (two-qubit) we generalise to the 16x16 two-qubit Pauli basis -> 256 real numbers.
"""
from __future__ import annotations
import numpy as np


_PAULI_1Q = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_basis(d: int) -> list[np.ndarray]:
    """Return d^2 Hermitian basis matrices (Pauli for d=2, two-qubit Pauli for d=4)."""
    if d == 2:
        return [_PAULI_1Q[c] for c in "IXYZ"]
    if d == 4:
        out = []
        for c1 in "IXYZ":
            for c2 in "IXYZ":
                out.append(np.kron(_PAULI_1Q[c1], _PAULI_1Q[c2]))
        return out
    raise ValueError(f"unsupported d={d}; only d in {{2,4}} implemented")


def choi_to_ptm(choi: np.ndarray, d: int) -> np.ndarray:
    """Convert a Choi matrix to a Pauli transfer matrix.

    Convention here: row-vectorisation Choi C = sum_i |i> <j| (x) Lambda(|i><j|),
    so that Lambda(rho) = Tr_1[(rho^T (x) I) C]. The resulting PTM entries are
        R[i, j] = (1/d) Tr[P_i Lambda(P_j)]
    which is real because every P_i is Hermitian and Lambda preserves Hermiticity.
    """
    paulis = _pauli_basis(d)
    n = len(paulis)
    R = np.zeros((n, n), dtype=float)
    for j, Pj in enumerate(paulis):
        # Reshape Choi to apply: Lambda(Pj) = Tr_1[(Pj^T (x) I) C]
        # Use the standard formula via 4-index reshape
        Cm = choi.reshape(d, d, d, d)
        # Lambda(rho) = sum_{m,n,p,q} C[m,p,n,q] rho[m,n] |p><q|
        out = np.einsum("mpnq,mn->pq", Cm, Pj)
        for i, Pi in enumerate(paulis):
            R[i, j] = float(np.real(np.trace(Pi @ out) / d))
    return R


def choi_to_ptm_features(choi: np.ndarray, d: int) -> np.ndarray:
    """PTM as a flat real feature vector of length d^4."""
    R = choi_to_ptm(choi, d)
    return R.reshape(-1)
