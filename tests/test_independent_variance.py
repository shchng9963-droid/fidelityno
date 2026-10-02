import numpy as np
import pytest
from physics.baselines.revision_measurement import (sample_measurement,
    calibration_outcome_variances, independent_measurement_variance, empirical_bayes_fusion)

def test_observation_return_preserves_stream_and_budget():
    ex = np.tile([1., .3, .3, 1.], (50,1))
    old = sample_measurement(ex,64,np.random.default_rng(8),'zz_known')
    new = sample_measurement(ex,64,np.random.default_rng(8),'zz_known',return_observations=True)
    for a,b in zip(old,new[:3]): np.testing.assert_array_equal(a,b)
    assert np.all(new[3]['setting_shots'].sum(1)==64)

def test_two_equal_outcomes_no_longer_zero_independent_proxy():
    obs = {'observed_means':np.ones((1,3)), 'setting_shots':np.full((1,3),2)}
    v = independent_measurement_variance([.6,.7,.8], obs['setting_shots'])
    assert v[0] > 0
    fused = empirical_bayes_fusion([.4],[1.],v,.02)
    assert .4 < fused[0] < 1

def test_three_shot_legacy_has_fixed_positive_variance():
    ex = np.tile([1., .2,.3,.4],(100,1))
    _,v,_ = sample_measurement(ex,3,np.random.default_rng(8))
    np.testing.assert_array_equal(v,np.full(100,3/16))

def test_known_z_formula_and_pooled_xy():
    p=.7; counts=np.full((3,2),10)
    np.testing.assert_allclose(independent_measurement_variance([4*p*(1-p)]*2,counts),p*(1-p)/20)
    observations={'observed_means':np.array([[.2,.6],[.4,.8]]),
                  'setting_shots':np.full((2,2),32),'allocation':'zz_known'}
    pooled=calibration_outcome_variances(observations)
    assert pooled[0]==pooled[1]

def test_invalid_calibration_counts_rejected():
    with pytest.raises(ValueError):
        calibration_outcome_variances({'observed_means':np.ones((2,3)),
            'setting_shots':np.ones((2,3)),'allocation':'fixed'})

def test_new_proxy_ignores_query_values_at_fixed_counts():
    a=sample_measurement(np.tile([1.,.2,.3,.4],(100,1)),8,np.random.default_rng(2),return_observations=True)
    b=sample_measurement(np.tile([1.,.8,.7,.6],(100,1)),8,np.random.default_rng(3),return_observations=True)
    np.testing.assert_array_equal(independent_measurement_variance([.7,.6,.5],a[3]['setting_shots']),
                                  independent_measurement_variance([.7,.6,.5],b[3]['setting_shots']))
