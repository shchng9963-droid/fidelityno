
from torch import nn
class PhysicsAuxHead(nn.Module):
    def __init__(self,d_model:int=256): super().__init__(); self.net=nn.Sequential(nn.Linear(d_model,d_model), nn.GELU(), nn.Linear(d_model,2))
    def forward(self,x): return self.net(x)
