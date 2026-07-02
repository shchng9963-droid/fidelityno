import numpy as np

from physics.composition import identity_channel


def test_identity_channel_ptm_is_identity_for_one_qubit():
    from physics.representations import choi_to_ptm, choi_to_features

    ch = identity_channel(2)
    ptm = choi_to_ptm(ch.choi, dim=2)

    assert ptm.shape == (4, 4)
    np.testing.assert_allclose(ptm, np.eye(4), atol=1e-8)
    features = choi_to_features(ch.choi, dim=2, mode="ptm")
    assert features.shape == (16,)
    np.testing.assert_allclose(features.reshape(4, 4), np.eye(4), atol=1e-8)


def test_representation_feature_shapes_for_two_qubit_identity():
    from physics.representations import choi_to_features

    ch = identity_channel(4)

    raw = choi_to_features(ch.choi, dim=4, mode="raw_choi_flat")
    herm = choi_to_features(ch.choi, dim=4, mode="choi_hermitian")
    ptm = choi_to_features(ch.choi, dim=4, mode="ptm")

    assert raw.shape == (512,)
    assert herm.shape == (512,)
    assert ptm.shape == (256,)
    assert raw.dtype == np.float32
    assert herm.dtype == np.float32
    assert ptm.dtype == np.float32


def test_unknown_representation_mode_raises():
    from physics.representations import choi_to_features

    ch = identity_channel(2)

    try:
        choi_to_features(ch.choi, dim=2, mode="does_not_exist")
    except ValueError as exc:
        assert "unknown channel representation" in str(exc)
    else:
        raise AssertionError("expected ValueError")
