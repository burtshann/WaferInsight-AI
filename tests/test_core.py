import numpy as np
import polars as pl
import pytest
from waferinsight.core import validate_map,clusters,map_records,yield_query,scan_records


def test_yield_excludes_outside():
    a=np.array([[0,1],[2,1]])
    r=yield_query(map_records(a).lazy()).collect().row(0,named=True)
    assert r['total_dies']==3 and r['good_dies']==2 and r['yield']==pytest.approx(2/3)


@pytest.mark.parametrize('a', [[[0,0],[0,0]], [[1,3],[0,1]], [[1,float('nan')],[1,1]], [1,2]])
def test_invalid_maps(a):
    with pytest.raises(ValueError): validate_map(a)


def test_clusters_four_connectivity():
    a=np.array([[2,2,0],[1,0,2],[1,1,2]])
    c=clusters(a,min_size=2)
    assert len(c)==2 and c[0]['dies']==2
    assert c[0]['x_min']==0 and c[0]['x_max']==1


def test_record_validation(tmp_path):
    d=map_records([[1,2],[1,0]])
    p=tmp_path/'d.csv'
    pl.concat([d,d]).write_csv(p)
    with pytest.raises(ValueError,match='Duplicate'): scan_records(p)
    d.with_columns(pl.lit(8).alias('status')).write_csv(p)
    with pytest.raises(ValueError,match='status'): scan_records(p)


def test_streaming_matches_memory(tmp_path):
    d=map_records([[1,2],[1,0]])
    p=tmp_path/'d.parquet'
    d.write_parquet(p)
    assert yield_query(scan_records(p)).collect(engine='streaming').equals(yield_query(d.lazy()).collect())
