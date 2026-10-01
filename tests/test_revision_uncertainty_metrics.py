import numpy as np
import pandas as pd
import torch
import eval as evaluation


def test_finite_grid_score_is_twice_mean_pinball():
    levels=np.arange(1,10)/10
    q=np.tile(levels,(2,1)); y=np.array([.2,.8])
    assert evaluation.quantile_score(q,y,levels)==2*evaluation.pinball_np(q,y,levels)


def test_evaluation_reports_length_specific_coverage(tmp_path,monkeypatch):
    levels=np.arange(1,10)/10
    class Fixed(torch.nn.Module):
        def forward(self,x,mask):
            return torch.tensor(levels,dtype=torch.float32).expand(len(x),-1),None
    checkpoint={'cfg':{'model':{'name':'fixed','quantiles':levels.tolist()},'seed':0},'model':{}}
    monkeypatch.setattr(evaluation.torch,'load',lambda *a,**k:checkpoint)
    monkeypatch.setattr(evaluation,'make_model',lambda *a,**k:Fixed())
    y=np.array([.05,.15,.1,.9],dtype=np.float32)
    path=tmp_path/'test.npz'
    np.savez(path,x=np.zeros((4,1,1)),mask=np.ones((4,1)),y=y,stats=np.zeros((4,2)),length=[1,1,2,2])
    evaluation.eval_ckpt('unused',{'test':str(path)},str(tmp_path/'result.csv'))
    result=pd.read_csv(tmp_path/'result.csv')
    q=np.tile(levels.astype(np.float32),(4,1))
    for length in [1,2]:
        idx=np.array([1,1,2,2])==length
        expected=evaluation.ece_quantile(q[idx],y[idx],levels)[0]
        assert np.isclose(result[result.length==length].ece.iloc[0],expected)
    assert result.ece.nunique()==2
