import numpy as np
from scripts.eval_revision_confirmatory import fit
from physics.baselines.revision_measurement import sample_measurement


def test_constants_fit_unclipped_calibration_labels():
    labels=np.array([-.25,.25,.5,1.])
    ids=np.arange(4)
    mean=fit('constant_mean',{}, {},ids,labels,np.arange(3))
    median=fit('constant_median',{}, {},ids,labels,np.arange(3))
    np.testing.assert_array_equal(mean,np.full(3,labels.mean()))
    np.testing.assert_array_equal(median,np.full(3,np.median(labels)))


def test_exchange_labels_remain_unclipped_and_variance_matches_outcomes():
    # r_X=r_Y=r_Z=-1/3 is a valid Pauli channel with zero identity weight.
    ex=np.tile([1.,-1/3,-1/3,-1/3],(4096,1))
    labels,var,_,detail=sample_measurement(ex,64,np.random.default_rng(101),
        'fixed',return_observations=True)
    assert np.any(labels<0)
    observed=detail['observed_means'];counts=detail['setting_shots']
    np.testing.assert_allclose(labels,.25+.25*observed.sum(1))
    np.testing.assert_allclose(var,((1-observed**2)/(counts-1)).sum(1)/16)


def test_small_pilot_label_draws_no_pilot_and_preserves_random_stream():
    ex=np.tile([1.,.2,-.1,.4],(10,1))
    for budget in [3,4,6,8,12]:
        fixed=sample_measurement(ex,budget,np.random.default_rng(17),'fixed',return_observations=True)
        pilot=sample_measurement(ex,budget,np.random.default_rng(17),'pilot',return_observations=True)
        for i in range(3):np.testing.assert_array_equal(fixed[i],pilot[i])
        np.testing.assert_array_equal(fixed[3]['setting_shots'],pilot[3]['setting_shots'])
