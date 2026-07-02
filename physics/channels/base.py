
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np

Array = np.ndarray


def dagger(x: Array) -> Array:
    return np.asarray(x).conj().T


def vec(x: Array) -> Array:
    return np.asarray(x).reshape(-1, order="F")


def unvec(v: Array, d: int) -> Array:
    return np.asarray(v).reshape((d, d), order="F")


def kraus_to_superop(kraus: list[Array]) -> Array:
    return sum(np.kron(k, k.conj()) for k in kraus)


def superop_to_choi(superop: Array, d: int) -> Array:
    choi = np.zeros((d*d, d*d), dtype=np.complex128)
    for i in range(d):
        for j in range(d):
            eij = np.zeros((d, d), dtype=np.complex128); eij[i, j] = 1.0
            out = unvec(superop @ vec(eij), d)
            choi[i*d:(i+1)*d, j*d:(j+1)*d] = out
    return choi


def choi_to_superop(choi: Array, d: int) -> Array:
    s = np.zeros((d*d, d*d), dtype=np.complex128)
    for i in range(d):
        for j in range(d):
            block = choi[i*d:(i+1)*d, j*d:(j+1)*d]
            col = i + j*d  # column-major vec(|i><j|)
            s[:, col] = vec(block)
    return s


def choi_to_real_features(choi: Array) -> Array:
    c = np.asarray(choi)
    herm = 0.5 * (c + c.conj().T)
    return np.concatenate([herm.real.reshape(-1), herm.imag.reshape(-1)]).astype(np.float32)


def partial_trace_output_choi(choi: Array, d: int) -> Array:
    # Our Choi block (i,j) is E(|i><j|). TP iff Tr[E(|i><j|)] = delta_ij.
    out = np.zeros((d, d), dtype=np.complex128)
    for i in range(d):
        for j in range(d):
            out[i, j] = np.trace(choi[i*d:(i+1)*d, j*d:(j+1)*d])
    return out

@dataclass
class Channel:
    name: str
    dim: int
    kraus: list[Array] | None = None
    choi: Array | None = None
    superop: Array | None = None
    params: Array = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.superop is None:
            if self.kraus is not None:
                self.superop = kraus_to_superop(self.kraus)
            elif self.choi is not None:
                self.superop = choi_to_superop(self.choi, self.dim)
            else:
                raise ValueError("Channel needs kraus, choi, or superop")
        if self.choi is None:
            self.choi = superop_to_choi(self.superop, self.dim)
        if self.kraus is None:
            self.kraus = []
        self.params = np.asarray(self.params, dtype=np.float64)

    def apply(self, rho: Array) -> Array:
        return unvec(self.superop @ vec(rho), self.dim)

    def compose_after(self, previous: "Channel") -> "Channel":
        if self.dim != previous.dim:
            raise ValueError("dimension mismatch")
        return Channel(
            name=f"{self.name}_after_{previous.name}", dim=self.dim,
            superop=self.superop @ previous.superop,
            params=np.concatenate([previous.params, self.params]) if previous.params.size or self.params.size else np.zeros(0),
            metadata={"factors": [previous.name, self.name]},
        )

    def is_cptp(self, atol: float = 1e-8) -> bool:
        herm = np.allclose(self.choi, self.choi.conj().T, atol=atol)
        evals = np.linalg.eigvalsh(0.5 * (self.choi + self.choi.conj().T))
        psd = bool(np.min(evals) >= -atol)
        tp = np.allclose(partial_trace_output_choi(self.choi, self.dim), np.eye(self.dim), atol=atol)
        return herm and psd and tp
