"""One system and two bath spins; fixed follow-up model, no neural fit."""
import numpy as np
from physics.channels.batched_memory import memory_grid_choi

I=np.eye(2); X=np.array([[0,1],[1,0]]); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1,-1])
def op(a,b,c): return np.kron(np.kron(a,b),c)
BATH=np.kron(np.diag([.8,.2]),np.diag([.65,.35]))
TERMS=np.array([(op(X,X,I)+op(Y,Y,I))/2,(op(X,I,X)+op(Y,I,Y))/2,
    op(Z,Z,I)/2,op(Z,I,Z)/2,op(I,Z,I)/2,op(I,I,Z)/2,
    (op(I,X,X)+op(I,Y,Y))/2])
# g1,g2,delta1,delta2,omega1,omega2,kappa,tau. Fixed before data generation.
LOW=np.array([.08,.08,.02,.02,.08,.08,.02,.25])
HIGH=np.array([.25,.25,.12,.12,.30,.30,.10,.90])
def unitary(p):
    e,v=np.linalg.eigh(np.einsum('k,kij->ij',p[:7],TERMS))
    return (v*np.exp(-1j*p[7]*e))@v.conj().T
def sample_parameters(rng,length): return rng.uniform(LOW,HIGH,size=(length,8))
def marginal_superop(u):
    j=memory_grid_choi([u],[0.],BATH)[0]
    # Choi indices are (input,output,input,output); row-major vectorisation.
    return j.reshape(2,2,2,2).transpose(1,3,0,2).reshape(4,4), j
def fidelity_from_superop(s): return float(np.trace(s).real/4)
