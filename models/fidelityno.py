
import torch
from torch import nn
from models.encoders.channel import ChoiEncoder
from models.encoders.state import StateTargetEncoder
from models.backbone import CausalTransformerBackbone
from models.heads.quantile import QuantileHead
from models.heads.gaussian import GaussianHead
from models.heads.scalar import ScalarHead
from models.heads.physics_aux import PhysicsAuxHead

class FidelityNO(nn.Module):
    def __init__(self,input_dim:int,d_model:int=256,depth:int=6,heads:int=8,head_type:str='quantile',causal:bool=True,aux:bool=True):
        super().__init__(); self.head_type=head_type; self.aux_enabled=aux
        self.channel_encoder=ChoiEncoder(input_dim,d_model)
        self.state_token=nn.Parameter(torch.zeros(1,1,d_model)); self.target_token=nn.Parameter(torch.zeros(1,1,d_model))
        self.pos=nn.Parameter(torch.randn(1,256,d_model)*0.01)
        self.backbone=CausalTransformerBackbone(d_model,heads,depth,causal=causal)
        
        if head_type=='quantile':
            self.head=QuantileHead(d_model)
        elif head_type=='gaussian':
            self.head=GaussianHead(d_model)
        elif head_type=='scalar':
            self.head=ScalarHead(d_model)
        else:
            raise ValueError(f'unknown head_type: {head_type}')
        self.aux=PhysicsAuxHead(d_model) if aux else None
    def forward(self,x,mask=None):
        B,L,_=x.shape; z=self.channel_encoder(x)
        st=self.state_token.expand(B,-1,-1); ot=self.target_token.expand(B,-1,-1)
        tokens=torch.cat([st,z,ot],dim=1)+self.pos[:,:L+2]
        key_padding_mask=None
        if mask is not None:
            key_padding_mask=torch.cat([torch.zeros(B,1,device=x.device,dtype=torch.bool), mask<0.5, torch.zeros(B,1,device=x.device,dtype=torch.bool)],dim=1)
        h=self.backbone(tokens,key_padding_mask=key_padding_mask)[:,-1]
        out=self.head(h)
        aux=self.aux(h) if self.aux is not None else None
        return out, aux
