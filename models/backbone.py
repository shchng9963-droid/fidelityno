
import torch
from torch import nn
class CausalTransformerBackbone(nn.Module):
    def __init__(self,d_model:int=256,nhead:int=8,num_layers:int=6,dropout:float=0.1,causal:bool=True):
        super().__init__(); self.causal=causal
        layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=4*d_model,dropout=dropout,batch_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(layer,num_layers=num_layers)
    def forward(self,tokens, key_padding_mask=None):
        L=tokens.size(1); mask=None
        if self.causal: mask=torch.triu(torch.ones(L,L,device=tokens.device,dtype=torch.bool),1)
        return self.encoder(tokens, mask=mask, src_key_padding_mask=key_padding_mask)
