import torch


def test_generic_gnn_forward_outputs_monotone_quantiles():
    from models.baselines.generic_gnn import GenericPathGNN

    model = GenericPathGNN(input_dim=6, d_model=16, layers=2, out=9)
    x = torch.randn(3, 5, 6)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        dtype=torch.float32,
    )

    q, aux = model(x, mask)

    assert aux is None
    assert q.shape == (3, 9)
    assert torch.all(q >= 0.0)
    assert torch.all(q <= 1.0)
    assert torch.all(q[:, 1:] >= q[:, :-1])


def test_generic_gnn_uses_raw_input_not_fidelityno_encoders():
    from models.baselines.generic_gnn import GenericPathGNN
    from models.encoders.channel import ChoiEncoder
    from models.heads.physics_aux import PhysicsAuxHead

    model = GenericPathGNN(input_dim=6, d_model=16, layers=2, out=9)
    modules = list(model.modules())

    assert not any(isinstance(m, ChoiEncoder) for m in modules)
    assert not any(isinstance(m, PhysicsAuxHead) for m in modules)


def test_train_factory_builds_generic_gnn():
    from omegaconf import OmegaConf
    from train import make_model
    from models.baselines.generic_gnn import GenericPathGNN

    cfg = OmegaConf.create({"model": {"name": "generic_gnn", "d_model": 16, "layers": 2}})
    model = make_model("generic_gnn", input_dim=6, max_len=5, cfg=cfg)

    assert isinstance(model, GenericPathGNN)
