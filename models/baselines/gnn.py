
import torch
from torch import nn
import torch.nn.functional as F


class LinearChainGNN(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 256, layers: int = 4, out: int = 9):
        super().__init__()
        self.inp = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([nn.Linear(3 * d_model, d_model) for _ in range(layers)])
        self.out = nn.Linear(d_model, out)
        # Anti-collapse init for monotone quantile head (see heads/quantile.py).
        with torch.no_grad():
            self.out.bias.zero_()
            if out > 1:
                self.out.bias[1:].fill_(-3.0)

    def forward(self, x, mask=None):
        h = torch.relu(self.inp(x))
        for layer in self.layers:
            left = torch.roll(h, 1, 1); left[:, 0] = 0
            right = torch.roll(h, -1, 1); right[:, -1] = 0
            h = torch.relu(layer(torch.cat([left, h, right], -1)))
            if mask is not None:
                h = h * mask.unsqueeze(-1)
        pooled = h.sum(1) / (mask.sum(1, keepdim=True) + 1e-8) if mask is not None else h.mean(1)
        raw = self.out(pooled)
        base = torch.sigmoid(raw[..., :1])
        if raw.shape[-1] > 1:
            increments = F.softplus(raw[..., 1:])
            q = torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)
        else:
            q = base
        return q.clamp(0.0, 1.0), None
