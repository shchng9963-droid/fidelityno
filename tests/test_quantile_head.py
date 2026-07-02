
import torch
from models.heads.quantile import QuantileHead

def test_quantile_head_monotonicity():
    head=QuantileHead(d_model=16)
    q=head(torch.randn(32,16))
    assert torch.all(q[:,1:] >= q[:,:-1] - 1e-7)
    assert torch.all((q>=0) & (q<=1))
