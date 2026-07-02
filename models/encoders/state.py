
import torch
from torch import nn
class StateTargetEncoder(nn.Module):
    def __init__(self, feature_dim: int, d_model: int=256):
        super().__init__(); self.net=nn.Sequential(nn.Linear(feature_dim,d_model), nn.GELU(), nn.Linear(d_model,d_model))
    def forward(self, x): return self.net(x)
