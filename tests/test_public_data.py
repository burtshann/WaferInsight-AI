import numpy as np
import pandas as pd
from scripts.prepare_public_benchmark import prepare


def test_public_preparation_removes_duplicate_and_conflicting_content(tmp_path):
    rows=[]
    # Distinct maps/lot IDs make it possible to verify all three splits.
    for i in range(20):
        a=np.ones((8,8),dtype=np.uint8)
        a.flat[i]=2
        rows.append({'waferMap':a,'failureType':np.array([['Center']]),'lotName':f'lot-{i}'})
    rows.append({**rows[0],'lotName':'duplicate-lot'})
    rows.append({**rows[1],'failureType':np.array([['Donut']])})
    rows.append({**rows[2],'failureType':np.empty((0,0))})
    source=tmp_path/'source.pkl'
    pd.DataFrame(rows).to_pickle(source)
    prepare(source,tmp_path/'out',tmp_path/'reports',train_cap=100,eval_cap=100)
    result=pd.read_csv(tmp_path/'out'/'manifest.csv')
    assert len(result)==19
    assert result.map_sha256.is_unique
    assert set(result.split)=={'train','validation','test'}
    assert result.groupby('lot_id').split.nunique().max()==1
    assert result.label.eq('Center').all()
