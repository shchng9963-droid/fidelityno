
import torch
from torch import nn
import torch.nn.functional as F


class GenericPathGNN(nn.Module):
    """Generic path-GNN baseline over raw per-channel vectors.

    This baseline intentionally avoids the FidelityNO Choi/state/target encoders
    and physics auxiliary head. It is used to test whether FidelityNO-G gains
    come from the full operator-aware framework or merely from path message
    passing over a channel sequence.
    """

    def __init__(self, input_dim: int, d_model: int = 256, layers: int = 4, out: int = 9):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(3 * d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )
                for _ in range(layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.readout = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out),
        )
        # Anti-collapse init for monotone quantile head (see heads/quantile.py).
        last = self.readout[-1]
        with torch.no_grad():
            last.bias.zero_()
            if out > 1:
                last.bias[1:].fill_(-3.0)

    def forward(self, x, mask=None):
        h = self.input_proj(x)
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        for layer, norm in zip(self.layers, self.norms):
            left = torch.roll(h, 1, dims=1)
            left[:, 0] = 0
            right = torch.roll(h, -1, dims=1)
            right[:, -1] = 0
            msg = layer(torch.cat([left, h, right], dim=-1))
            h = norm(h + msg)
            if mask is not None:
                h = h * mask.unsqueeze(-1)
        if mask is not None:
            denom = mask.sum(1, keepdim=True).clamp_min(1.0)
            mean_pool = h.sum(1) / denom
            last_idx = (denom.squeeze(1).long() - 1).clamp_min(0)
            last = h[torch.arange(h.shape[0], device=h.device), last_idx]
        else:
            mean_pool = h.mean(1)
            last = h[:, -1]
        raw = self.readout(torch.cat([mean_pool, last], dim=-1))
        base = torch.sigmoid(raw[..., :1])
        if raw.shape[-1] > 1:
            increments = F.softplus(raw[..., 1:])
            q = torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)
        else:
            q = base
        return q.clamp(0.0, 1.0), None
