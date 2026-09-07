"""Stream per-map die records from a labeled manifest into CSV."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from waferinsight.core import map_records


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('data/die_records.csv'))
    a=p.parse_args()
    frame=pd.read_csv(a.manifest,dtype=str)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8',newline='') as f:
        for i,r in frame.iterrows():
            records=map_records(np.load(a.manifest.parent/r.map_path,allow_pickle=False),str(i),r.lot_id)
            f.write(records.write_csv(include_header=i==0))
    print(f'Exported {len(frame)} wafers to {a.output}')


if __name__=='__main__':
    main()
