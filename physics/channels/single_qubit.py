
import numpy as np
from .base import Channel

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0,1],[1,0]], dtype=np.complex128)
Y = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
Z = np.array([[1,0],[0,-1]], dtype=np.complex128)


def amplitude_damping(gamma: float) -> Channel:
    gamma = float(np.clip(gamma, 0.0, 1.0))
    k0 = np.array([[1,0],[0,np.sqrt(1-gamma)]], dtype=np.complex128)
    k1 = np.array([[0,np.sqrt(gamma)],[0,0]], dtype=np.complex128)
    return Channel("amplitude_damping", 2, kraus=[k0,k1], params=np.array([gamma]))


def phase_damping(lambda_p: float) -> Channel:
    l = float(np.clip(lambda_p, 0.0, 1.0))
    k0 = np.sqrt(1-l) * I2
    k1 = np.sqrt(l) * np.array([[1,0],[0,0]], dtype=np.complex128)
    k2 = np.sqrt(l) * np.array([[0,0],[0,1]], dtype=np.complex128)
    return Channel("phase_damping", 2, kraus=[k0,k1,k2], params=np.array([l]))


def depolarizing(p: float) -> Channel:
    p = float(np.clip(p, 0.0, 1.0))
    return pauli_channel(p/3, p/3, p/3, name="depolarizing")


def pauli_channel(p_x: float, p_y: float, p_z: float, name: str = "pauli") -> Channel:
    px, py, pz = [max(0.0, float(x)) for x in (p_x,p_y,p_z)]
    s = px + py + pz
    if s > 1.0:
        px, py, pz = px/s, py/s, pz/s
    p0 = max(0.0, 1.0-px-py-pz)
    kraus = [np.sqrt(p0)*I2, np.sqrt(px)*X, np.sqrt(py)*Y, np.sqrt(pz)*Z]
    return Channel(name, 2, kraus=kraus, params=np.array([px,py,pz]))


def sample_single_qubit(rng: np.random.Generator, family: str | None = None) -> Channel:
    fams = ["amplitude_damping", "phase_damping", "depolarizing", "pauli"]
    family = family or rng.choice(fams)
    if family == "amplitude_damping": return amplitude_damping(rng.uniform(0, 0.25))
    if family == "phase_damping": return phase_damping(rng.uniform(0, 0.3))
    if family == "depolarizing": return depolarizing(rng.uniform(0, 0.2))
    if family == "pauli":
        probs = rng.dirichlet([1,1,1]) * rng.uniform(0, 0.25)
        return pauli_channel(*probs)
    raise ValueError(f"unknown family {family}")
