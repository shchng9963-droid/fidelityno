import numpy as np
from physics.channels.two_bath_memory import BATH,unitary,sample_parameters,marginal_superop,fidelity_from_superop
from physics.channels.batched_memory import memory_grid_choi,memory_grid_fidelity

def test_reset_composition_and_cptp():
    rng=np.random.default_rng(104)
    us=[unitary(p) for p in sample_parameters(rng,8)]
    s=np.eye(4,dtype=complex)
    for u in us:
        step,j=marginal_superop(u);s=step@s
        np.testing.assert_allclose(np.trace(j.reshape(2,2,2,2),axis1=1,axis2=3),np.eye(2),atol=1e-12)
        assert np.linalg.eigvalsh(j).min()>-1e-12
    np.testing.assert_allclose(memory_grid_fidelity(us,[0.],BATH)[0],fidelity_from_superop(s),atol=1e-12)
    for j in memory_grid_choi(us,[0,.9,1],BATH):
        assert np.linalg.eigvalsh(j).min()>-1e-12
        np.testing.assert_allclose(np.trace(j.reshape(2,2,2,2),axis1=1,axis2=3),np.eye(2),atol=1e-12)
def test_identity():
    np.testing.assert_allclose(memory_grid_fidelity([np.eye(8)]*4,[0,.5,1],BATH),1,atol=1e-12)
def test_spectator_bath_matches_single_bath():
    from physics.channels.collision_exchange_memory import _exchange_unitary,DEFAULT_BATH
    one=[_exchange_unitary(.15,.05,.2,.6)]*5
    two=[np.kron(u,np.eye(2)) for u in one]
    np.testing.assert_allclose(memory_grid_fidelity(one,[0,.9,1],DEFAULT_BATH),
        memory_grid_fidelity(two,[0,.9,1],BATH),atol=1e-12)
