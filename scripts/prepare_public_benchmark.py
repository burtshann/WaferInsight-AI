"""Audit trusted WM-811K and create a deterministic, lot-disjoint research subset."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from waferinsight.core import PATTERNS,validate_map,map_records


def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def prepare(source,output,report,train_cap=300,eval_cap=100):
    print('Loading trusted public pickle',flush=True)
    raw=pd.read_pickle(source)
    if raw.lotName.isna().any():raise ValueError('Missing lot IDs in source data')
    labels=[str(v[0]) if len(v:=np.asarray(x).ravel())==1 else '' for x in raw.failureType]
    counts=Counter(labels)
    frame=pd.DataFrame({'source_index':np.arange(len(raw)),'label':labels,'lot_id':raw.lotName.astype(str).to_numpy()})
    frame=frame[frame.label.isin(PATTERNS)].copy()
    print('Raw',len(raw),'labeled',len(frame),flush=True)
    hashes=[]
    for i in frame.source_index:
        a=validate_map(raw.iloc[i].waferMap)
        hashes.append(hashlib.sha256(np.asarray(a.shape,dtype='<i4').tobytes()+a.tobytes()).hexdigest())
    frame['map_sha256']=hashes
    before=len(frame)
    # Conflicting exact-content labels are excluded entirely; otherwise retain first source row.
    conflicts=frame.groupby('map_sha256').label.nunique()
    conflict_keys=set(conflicts[conflicts>1].index)
    conflicting_rows=int(frame.map_sha256.isin(conflict_keys).sum())
    frame=frame[~frame.map_sha256.isin(conflict_keys)].drop_duplicates('map_sha256').copy()
    lots=np.array(sorted(frame.lot_id.unique()))
    np.random.default_rng(42).shuffle(lots)
    n=max(1,int(len(lots)*.15))
    parts=[]
    for split,group,cap in [('train',lots[2*n:],train_cap),('validation',lots[:n],eval_cap),('test',lots[n:2*n],eval_cap)]:
        f=frame[frame.lot_id.isin(group)]
        for label in PATTERNS:
            sub=f[f.label==label]
            if len(sub):
                chosen=sub.sample(n=min(cap,len(sub)),random_state=42).copy()
                chosen['split']=split
                parts.append(chosen)
    selected=pd.concat(parts).sort_values('source_index').reset_index(drop=True)
    (output/'maps').mkdir(parents=True,exist_ok=True)
    selected['map_path']=[f'maps/{i:07d}.npy' for i in selected.source_index]
    die_rows=0
    # ETL input is the first 1,000 selected source rows, with original die maps/coordinates.
    with (output/'die_records.csv').open('w',encoding='utf-8',newline='') as csv:
        for j,r in selected.iterrows():
            a=validate_map(raw.iloc[int(r.source_index)].waferMap)
            np.save(output/r.map_path,a,allow_pickle=False)
            if j<1000:
                records=map_records(a,str(r.source_index),r.lot_id)
                die_rows+=len(records)
                csv.write(records.write_csv(include_header=j==0))
    selected.to_csv(output/'manifest.csv',index=False)
    report.mkdir(parents=True,exist_ok=True)
    selected.to_csv(report/'split_manifest.csv',index=False)
    audit={'dataset':'WM-811K','source':'https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map',
      'version':1,'license_as_listed':'CC0: Public Domain','source_file_sha256':sha_file(source),
      'raw_wafers':len(raw),'raw_lots':int(raw.lotName.nunique()),'label_counts':dict(counts),
      'labeled_wafers':before,'conflicting_duplicate_rows_removed':conflicting_rows,
      'duplicate_or_conflicting_rows_removed':before-len(frame),'unique_labeled_wafers':len(frame),
      'selected_wafers':len(selected),'seed':42,'train_cap_per_class':train_cap,'eval_cap_per_class':eval_cap,
      'split_class_counts':{s:dict(Counter(f.label)) for s,f in selected.groupby('split')},
      'split_lots':{s:int(f.lot_id.nunique()) for s,f in selected.groupby('split')},
      'etl_wafers':min(1000,len(selected)),'etl_die_records':die_rows,
      'protocol':'Remove exact-map duplicates and conflicting labels before seed-42 lot split (~70/15/15); class-cap sampling inside each split. Not the official WM-811K split; subset class priors are altered.',
      'split_manifest_sha256':sha_file(report/'split_manifest.csv')}
    (report/'dataset_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('data/wm811k-public'))
    p.add_argument('--report',type=Path,default=Path('reports/wm811k'))
    p.add_argument('--trust-pickle',action='store_true')
    a=p.parse_args()
    if not a.trust_pickle:p.error('Use --trust-pickle only for the trusted public source; pickle can execute code.')
    prepare(a.input,a.output,a.report)
