
import torch
from torch import nn
import torch.nn.functional as F


class DeepSets(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 256, out: int = 9):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(input_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.rho = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, out))
        # Anti-collapse init for monotone quantile head (see heads/quantile.py).
        last = self.rho[-1]
        with torch.no_grad():
            last.bias.zero_()
            if out > 1:
                last.bias[1:].fill_(-3.0)

    def forward(self, x, mask=None):
        h = self.phi(x)
        if mask is not None:
            h = h * mask.unsqueeze(-1)
            pooled = h.sum(1) / (mask.sum(1, keepdim=True) + 1e-8)
        else:
            pooled = h.mean(1)
        raw = self.rho(pooled)
        base = torch.sigmoid(raw[..., :1])
        if raw.shape[-1] > 1:
            increments = F.softplus(raw[..., 1:])
            q = torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)
        else:
            q = base
        return q.clamp(0.0, 1.0), None
