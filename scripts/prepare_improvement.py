"""Expand original training lots; reserve untouched lots for an additional audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from waferinsight.core import PATTERNS,validate_map
from scripts.prepare_public_benchmark import sha_file


def prepare(source,old_manifest,out,report):
    expected='1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65'
    if sha_file(source)!=expected:raise ValueError('Unexpected public source checksum')
    raw=pd.read_pickle(source)
    old=pd.read_csv(old_manifest,dtype={'lot_id':str})
    labels=[str(v[0]) if len(v:=np.asarray(x).ravel())==1 else '' for x in raw.failureType]
    f=pd.DataFrame({'source_index':np.arange(len(raw)),'label':labels,'lot_id':raw.lotName.astype(str).to_numpy()})
    f=f[f.label.isin(PATTERNS)].copy()
    hashes=[]
    for i in f.source_index:
        a=validate_map(raw.iloc[i].waferMap)
        hashes.append(hashlib.sha256(np.asarray(a.shape,dtype='<i4').tobytes()+a.tobytes()).hexdigest())
    f['map_sha256']=hashes
    n=f.groupby('map_sha256').label.nunique()
    f=f[~f.map_sha256.isin(n[n>1].index)].drop_duplicates('map_sha256')
    lots=np.array(sorted(f.lot_id.unique()))
    np.random.default_rng(42).shuffle(lots)
    cut=int(len(lots)*.15)
    # Reconstruct the original complete lot allocation, not merely the capped subset.
    train_pool=f[f.lot_id.isin(lots[2*cut:])]
    parts=[]
    for label in PATTERNS:
        pool=train_pool[train_pool.label==label]
        cap=4000 if label=='none' else 2000
        selected=pool.sample(min(cap,len(pool)),random_state=20260909).copy()
        selected['split']='train'
        parts.append(selected)
    for name in ['validation','test']:
        selected=f[f.source_index.isin(old[old.split==name].source_index)].copy()
        selected['split']='validation' if name=='validation' else 'historical_test'
        parts.append(selected)
    fresh=f[f.lot_id.isin(lots[cut:2*cut])&~f.lot_id.isin(old.lot_id)]
    for label in PATTERNS:
        pool=fresh[fresh.label==label]
        if len(pool):
            selected=pool.sample(min(200,len(pool)),random_state=20260909).copy()
            selected['split']='fresh_audit'
            parts.append(selected)
    manifest=pd.concat(parts).sort_values('source_index').reset_index(drop=True)
    assert manifest.map_sha256.is_unique
    assert manifest.groupby('lot_id').split.nunique().max()==1
    out.mkdir(parents=True,exist_ok=True)
    report.mkdir(parents=True,exist_ok=True)
    # Cache compact categorical maps; nearest-neighbour resampling matches original ViT.
    cache=np.empty((len(manifest),64,64),dtype=np.uint8)
    for j,r in manifest.iterrows():
        a=validate_map(raw.iloc[int(r.source_index)].waferMap)
        yi=np.minimum((np.arange(64)*a.shape[0]/64).astype(int),a.shape[0]-1)
        xi=np.minimum((np.arange(64)*a.shape[1]/64).astype(int),a.shape[1]-1)
        cache[j]=a[yi[:,None],xi[None,:]]
    np.save(out/'maps.npy',cache,allow_pickle=False)
    manifest['cache_index']=np.arange(len(manifest))
    manifest.to_csv(out/'manifest.csv',index=False,lineterminator='\n')
    manifest.to_csv(report/'split_manifest.csv',index=False,lineterminator='\n')
    audit={'source_sha256':expected,'seed':20260909,
      'split_counts':manifest.groupby('split').size().to_dict(),
      'class_counts':{s:g.label.value_counts().to_dict() for s,g in manifest.groupby('split')},
      'fresh_missing_classes':sorted(set(PATTERNS)-set(manifest[manifest.split=='fresh_audit'].label)),
      'manifest_sha256':sha_file(report/'split_manifest.csv'),'cache_sha256':sha_file(out/'maps.npy'),
      'protocol':'Expanded original training lots (none cap4000, others cap2000). Original validation809 unchanged. Old test774 historical only. Fresh audit: unused lots in original full test partition, cap200/class; excluded ALL previously selected lots. Exact-content deduplicated. Class caps change priors.'}
    (report/'dataset_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--trust-pickle',action='store_true')
    p.add_argument('--old-manifest',type=Path,default=Path('reports/wm811k/split_manifest.csv'))
    p.add_argument('--output',type=Path,default=Path('data/improvement'))
    p.add_argument('--report',type=Path,default=Path('reports/improvement'))
    a=p.parse_args()
    if not a.trust_pickle:p.error('Trusted source pickle requires --trust-pickle')
    prepare(a.input,a.old_manifest,a.output,a.report)
