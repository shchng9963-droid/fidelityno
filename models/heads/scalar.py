import torch
from torch import nn


class ScalarHead(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):
        return torch.sigmoid(self.proj(x))
