
import torch
from torch import nn
import torch.nn.functional as F


class FlatMLP(nn.Module):
    def __init__(self, input_dim: int, max_len: int, d_model: int = 256, out: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * max_len, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, out),
        )
        # Anti-collapse init for monotone quantile head (see heads/quantile.py).
        last = self.net[-1]
        with torch.no_grad():
            last.bias.zero_()
            if out > 1:
                last.bias[1:].fill_(-3.0)

    def forward(self, x, mask=None):
        raw = self.net(x)
        base = torch.sigmoid(raw[..., :1])
        if raw.shape[-1] > 1:
            increments = F.softplus(raw[..., 1:])
            q = torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)
        else:
            q = base
        return q.clamp(0.0, 1.0), None
