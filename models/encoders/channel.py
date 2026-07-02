
import torch
from torch import nn

class ChoiEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int=256, hidden: int=512):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, d_model))
    def forward(self, x):
        return self.net(x)

class PTMEncoder(ChoiEncoder):
    """Ablation hook: same API; data pipeline can feed PTM features later."""
    pass
