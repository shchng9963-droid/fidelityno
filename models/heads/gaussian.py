
import torch
from torch import nn
class GaussianHead(nn.Module):
    def __init__(self, d_model:int=256): super().__init__(); self.proj=nn.Linear(d_model,2)
    def forward(self,x):
        y=self.proj(x); mu=torch.sigmoid(y[...,0]); sigma=torch.nn.functional.softplus(y[...,1])+1e-4; return mu,sigma

def truncated_normal_nll(mu, sigma, target):
    dist=torch.distributions.Normal(mu,sigma)
    z=(dist.cdf(torch.ones_like(mu))-dist.cdf(torch.zeros_like(mu))).clamp_min(1e-8)
    return (-(dist.log_prob(target)-torch.log(z))).mean()
