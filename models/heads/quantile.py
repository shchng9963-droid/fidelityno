
import torch
from torch import nn
import torch.nn.functional as F


class QuantileHead(nn.Module):
    """Monotone quantile head: base quantile via sigmoid + softplus increments.

    Each quantile can freely range in [0, 1] while maintaining monotonicity.
    Unlike cumsum/sum normalization, the highest quantile is NOT forced to 1.0.

    Init: bias of increment outputs is set negative so softplus(bias) is small,
    keeping initial quantiles in a tight band around 0.5 instead of clamping
    to 1.0 (which kills gradients through the increment branch and causes a
    "predict constant mean" collapse).
    """

    def __init__(self, d_model: int = 256, levels=None):
        super().__init__()
        self.levels = torch.tensor(
            levels or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            dtype=torch.float32,
        )
        n_levels = len(self.levels)
        # Output: 1 base logit + (n_levels - 1) increment logits
        self.proj = nn.Linear(d_model, n_levels)
        # Anti-collapse init: increment bias = -3 so softplus(-3) ~= 0.049,
        # giving an initial quantile spread of ~0.4 across all 9 levels.
        # Base bias = 0 keeps median initial output near 0.5.
        with torch.no_grad():
            self.proj.bias.zero_()
            if n_levels > 1:
                self.proj.bias[1:].fill_(-3.0)

    def forward(self, x):
        raw = self.proj(x)  # [B, n_levels]
        # First quantile: sigmoid maps to [0, 1]
        base = torch.sigmoid(raw[..., :1])  # [B, 1]
        # Remaining quantiles: non-negative increments via softplus
        if raw.shape[-1] > 1:
            increments = F.softplus(raw[..., 1:])  # [B, n_levels-1]
            # Cumulative sum of increments added to base
            q = torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)
        else:
            q = base
        return q.clamp(0.0, 1.0)


def pinball_loss(pred, target, levels):
    q = levels.to(pred.device).view(1, -1)
    e = target.view(-1, 1) - pred
    return torch.maximum(q * e, (q - 1) * e).mean()


def quantile_mean(q):
    return q.mean(dim=-1)
