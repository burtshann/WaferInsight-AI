import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from waferinsight.predict import load_inspector,inspect_map,DEFAULT_MODEL


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('input',type=Path,help='JSON array, headerless CSV or NPY map')
    p.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    a=p.parse_args()
    if a.input.suffix=='.npy':wafer=np.load(a.input,allow_pickle=False)
    elif a.input.suffix=='.json':wafer=json.loads(a.input.read_text())
    else:wafer=np.loadtxt(a.input,delimiter=',')
    model,_=load_inspector(a.model)
    print(json.dumps(inspect_map(wafer,model),indent=2))
