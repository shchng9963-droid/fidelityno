import torch
from omegaconf import OmegaConf


def test_fidelityno_scalar_head_trains_with_mse_helpers():
    from models.fidelityno import FidelityNO
    from train import loss_for_prediction, mean_from_prediction, prediction_to_quantiles

    levels = torch.tensor([0.1, 0.5, 0.9])
    model = FidelityNO(input_dim=8, d_model=16, depth=1, heads=2, head_type="scalar", aux=False)
    x = torch.randn(4, 3, 8)
    mask = torch.ones(4, 3)
    y = torch.rand(4)

    pred, aux = model(x, mask)
    loss = loss_for_prediction(pred, y, levels, "scalar")
    mean = mean_from_prediction(pred)
    q = prediction_to_quantiles(pred, levels)

    assert aux is None
    assert pred.shape == (4, 1)
    assert loss.ndim == 0
    assert mean.shape == (4,)
    assert q.shape == (4, 3)
    assert torch.all(q[:, 0] == q[:, 1])


def test_fidelityno_gaussian_head_has_positive_sigma_and_quantiles():
    from models.fidelityno import FidelityNO
    from train import loss_for_prediction, mean_from_prediction, prediction_to_quantiles

    levels = torch.tensor([0.1, 0.5, 0.9])
    model = FidelityNO(input_dim=8, d_model=16, depth=1, heads=2, head_type="gaussian", aux=False)
    x = torch.randn(4, 3, 8)
    y = torch.rand(4)

    pred, _ = model(x, torch.ones(4, 3))
    mu, sigma = pred
    loss = loss_for_prediction(pred, y, levels, "gaussian")
    mean = mean_from_prediction(pred)
    q = prediction_to_quantiles(pred, levels)

    assert mu.shape == (4,)
    assert sigma.shape == (4,)
    assert torch.all(sigma > 0)
    assert loss.ndim == 0
    assert mean.shape == (4,)
    assert q.shape == (4, 3)
    assert torch.all(q[:, 1:] >= q[:, :-1])
    assert torch.all(q >= 0.0)
    assert torch.all(q <= 1.0)


def test_train_factory_passes_head_type_to_fidelityno():
    from train import make_model
    from models.heads.scalar import ScalarHead

    cfg = OmegaConf.create({"model": {"name": "fidelityno", "d_model": 16, "depth": 1, "heads": 2, "head_type": "scalar", "aux": False}})
    model = make_model("fidelityno", input_dim=8, max_len=3, cfg=cfg)

    assert isinstance(model.head, ScalarHead)
