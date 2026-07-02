from __future__ import annotations

import itertools
import numpy as np

from physics.channels.base import choi_to_real_features, choi_to_superop, unvec, vec


def raw_choi_features(choi: np.ndarray) -> np.ndarray:
    c = np.asarray(choi)
    return np.concatenate([c.real.reshape(-1), c.imag.reshape(-1)]).astype(np.float32)


def hermitian_choi_features(choi: np.ndarray) -> np.ndarray:
    return choi_to_real_features(choi)


def pauli_basis(dim: int) -> list[np.ndarray]:
    if dim < 1:
        raise ValueError(f"PTM representation requires power-of-two Hilbert dimension, got {dim}")
    n_float = np.log2(dim)
    n = int(round(float(n_float)))
    if 2 ** n != dim:
        raise ValueError(f"PTM representation requires power-of-two Hilbert dimension, got {dim}")
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    single = [I, X, Y, Z]
    out = []
    for labels in itertools.product(range(4), repeat=n):
        m = np.array([[1]], dtype=np.complex128)
        for idx in labels:
            m = np.kron(m, single[idx])
        out.append(m)
    return out


def choi_to_ptm(choi: np.ndarray, dim: int) -> np.ndarray:
    superop = choi_to_superop(np.asarray(choi), dim)
    basis = pauli_basis(dim)
    ptm = np.zeros((dim * dim, dim * dim), dtype=np.float64)
    for j, bj in enumerate(basis):
        out = unvec(superop @ vec(bj), dim)
        for i, bi in enumerate(basis):
            ptm[i, j] = (np.trace(bi.conj().T @ out) / dim).real
    return ptm


def ptm_features(choi: np.ndarray, dim: int) -> np.ndarray:
    return choi_to_ptm(choi, dim).reshape(-1).astype(np.float32)


def compressed_hermitian_features(choi: np.ndarray) -> np.ndarray:
    """Compressed Hermitian Choi representation.

    Exploits the fact that a Choi matrix is exactly Hermitian (CPTP) so the
    raw real+imaginary flatten contains 2*N^2 numbers but only N^2 real DoF.
    Encoding: real diagonal (N) + real upper triangle (N(N-1)/2) +
    imag strict upper triangle (N(N-1)/2) = N^2 numbers, where N = d^2.
    """
    c = np.asarray(choi)
    # numerically symmetrize against floating-point drift before extracting
    h = 0.5 * (c + c.conj().T)
    N = h.shape[0]
    iu = np.triu_indices(N, k=1)
    diag = h.real.diagonal()                 # N reals
    re_upper = h[iu].real                    # N(N-1)/2 reals
    im_upper = h[iu].imag                    # N(N-1)/2 reals
    return np.concatenate([diag, re_upper, im_upper]).astype(np.float32)


def choi_to_features(choi: np.ndarray, dim: int, mode: str = "choi_hermitian") -> np.ndarray:
    if mode in {"choi_hermitian", "hermitian", "choi"}:
        return hermitian_choi_features(choi)
    if mode in {"raw_choi_flat", "raw"}:
        return raw_choi_features(choi)
    if mode in {"compressed_hermitian", "compressed"}:
        return compressed_hermitian_features(choi)
    if mode == "ptm":
        return ptm_features(choi, dim)
    raise ValueError(f"unknown channel representation mode: {mode}")


def feature_dim_for_representation(dim: int, mode: str = "choi_hermitian") -> int:
    N = dim * dim
    if mode in {"choi_hermitian", "hermitian", "choi", "raw_choi_flat", "raw"}:
        return 2 * N * N
    if mode in {"compressed_hermitian", "compressed"}:
        return N * N  # diag(N) + 2 * N(N-1)/2 = N^2
    if mode == "ptm":
        return N * N
    raise ValueError(f"unknown channel representation mode: {mode}")
