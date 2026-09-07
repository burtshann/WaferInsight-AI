"""Only deserialize a trusted local WM811K pickle: pickle can execute code."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from waferinsight.core import PATTERNS,validate_map


def main():
    p=argparse.ArgumentParser()
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path,default=Path('data/wm811k'))
    p.add_argument('--trust-pickle',action='store_true')
    a=p.parse_args()
    if not a.trust_pickle:
        p.error('Pickle can execute arbitrary code. Use --trust-pickle only for a trusted source.')
    df=pd.read_pickle(a.input)
    (a.output/'maps').mkdir(parents=True,exist_ok=True)
    manifest=[]
    for i,(_,r) in enumerate(df.iterrows()):
        labels=np.asarray(r.failureType).flatten()
        if len(labels)!=1 or str(labels[0]) not in PATTERNS:
            continue  # unlabeled is not the healthy none class
        if pd.isna(r.lotName):
            raise ValueError('Missing lotName; cannot construct leakage-safe split')
        wafer=validate_map(r.waferMap)
        name=f'maps/{i:07d}.npy'
        np.save(a.output/name,wafer,allow_pickle=False)
        manifest.append({'map_path':name,'label':str(labels[0]),'lot_id':str(r.lotName)})
    pd.DataFrame(manifest,columns=['map_path','label','lot_id']).to_csv(a.output/'manifest.csv',index=False)
    print(f'Exported {len(manifest)} labeled maps; no data uploaded.')


if __name__=='__main__':
    main()
