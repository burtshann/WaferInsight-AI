import numpy as np
import pandas as pd
import pytest
torch=pytest.importorskip('torch')
from waferinsight.model import WaferViT,preprocess
from scripts.train_vit import split_lots


def test_model_backward():
    model=WaferViT()
    x=preprocess(np.array([[0,1],[1,2]]))[None]
    out=model(x)
    assert out.shape==(1,9) and torch.isfinite(out).all()
    out.sum().backward()
    assert model.patch.weight.grad is not None


def test_lots_disjoint():
    frame=pd.DataFrame({'lot_id':['a','a','b','c','d','e']})
    parts=split_lots(frame)
    sets=[set(p.lot_id) for p in parts]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    assert sum(map(len,parts))==len(frame)
