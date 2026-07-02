
import numpy as np
from scipy.linalg import expm
from .base import Channel, vec, unvec
from .single_qubit import X, Y, Z, I2

def lindblad_superoperator(H: np.ndarray, jumps: list[np.ndarray]) -> np.ndarray:
    d=H.shape[0]; I=np.eye(d,dtype=np.complex128)
    L = -1j*(np.kron(I,H) - np.kron(H.T,I))
    for c in jumps:
        cd_c = c.conj().T @ c
        L += np.kron(c.conj(), c) - 0.5*np.kron(I, cd_c) - 0.5*np.kron(cd_c.T, I)
    return L

def lindblad_channel(H: np.ndarray, jumps: list[np.ndarray], t: float, params=None) -> Channel:
    d=H.shape[0]
    S=expm(lindblad_superoperator(H,jumps)*float(t))
    return Channel("lindblad", d, superop=S, params=np.asarray(params if params is not None else [t], float))

def sample_lindblad(rng: np.random.Generator) -> Channel:
    wx, wz = rng.uniform(-0.5,0.5), rng.uniform(-0.5,0.5)
    gd, gp = rng.uniform(0,0.15), rng.uniform(0,0.12)
    t = rng.uniform(0.05,1.0)
    sm = np.array([[0,1],[0,0]], dtype=np.complex128)
    H = 0.5*(wx*X + wz*Z)
    jumps=[]
    if gd>0: jumps.append(np.sqrt(gd)*sm)
    if gp>0: jumps.append(np.sqrt(gp)*Z)
    return lindblad_channel(H,jumps,t,params=np.array([wx,wz,gd,gp,t]))
