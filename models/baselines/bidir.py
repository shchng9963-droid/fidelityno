
from models.fidelityno import FidelityNO
class BidirectionalTransformer(FidelityNO):
    def __init__(self,*args,**kwargs): kwargs['causal']=False; super().__init__(*args,**kwargs)
