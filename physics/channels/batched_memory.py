"""Reference-state propagation over a batch of bath-retention values."""
import numpy as np


def memory_grid_choi(unitaries, eta_values, bath_state):
    eta = np.asarray(eta_values, dtype=float)
    if eta.ndim != 1 or len(eta) == 0 or np.any((eta < 0) | (eta > 1)):
        raise ValueError("retention values must be in [0,1]")
    phi = np.array([1,0,0,1], dtype=complex)/np.sqrt(2)
    bath_state = np.asarray(bath_state, dtype=complex)
    db = bath_state.shape[0]
    if bath_state.shape != (db, db):
        raise ValueError("bath state must be square")
    initial = np.kron(np.outer(phi, phi.conj()), bath_state)
    joint = np.broadcast_to(initial, (len(eta),4*db,4*db)).copy()
    weight = eta[:,None,None]
    for u in unitaries:
        full = np.kron(np.eye(2), u)
        joint = full @ joint @ full.conj().T
        rs = np.trace(joint.reshape(-1,4,db,4,db), axis1=2, axis2=4)
        reset = np.einsum("gij,ab->giajb", rs, bath_state).reshape(-1,4*db,4*db)
        joint = weight*joint + (1-weight)*reset
    return 2*np.trace(joint.reshape(-1,4,db,4,db), axis1=2, axis2=4)


def memory_grid_fidelity(unitaries, eta_values, bath_state):
    choi = memory_grid_choi(unitaries, eta_values, bath_state)
    return (choi[:,0,0]+choi[:,0,3]+choi[:,3,0]+choi[:,3,3]).real/4
