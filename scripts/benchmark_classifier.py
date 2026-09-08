"""Handcrafted spatial-feature Random Forest baseline on the same public split."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from waferinsight.core import PATTERNS,validate_map


def spatial_features(a):
    a=validate_map(a)
    y,x=np.indices(a.shape)
    x=(x+.5)/a.shape[1]*2-1
    y=(y+.5)/a.shape[0]*2-1
    valid=a>0
    fail=a==2
    radial=np.hypot(x,y)
    angle=np.arctan2(y,x)
    def density(mask):
        return float((fail&mask).sum()/max(1,(valid&mask).sum()))
    values=[density(valid)]
    for lo,hi in zip(np.linspace(0,1.5,11)[:-1],np.linspace(0,1.5,11)[1:]):
        values.append(density((radial>=lo)&(radial<hi)))
    for lo,hi in zip(np.linspace(-np.pi,np.pi,13)[:-1],np.linspace(-np.pi,np.pi,13)[1:]):
        values.append(density((angle>=lo)&(angle<hi)))
    for gy in range(8):
        for gx in range(8):
            values.append(density((x>=gx/4-1)&(x<(gx+1)/4-1)&(y>=gy/4-1)&(y<(gy+1)/4-1)))
    return values


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('reports/wm811k/baseline.json'))
    a=p.parse_args()
    f=pd.read_csv(a.manifest,dtype=str)
    x=np.array([spatial_features(np.load(a.manifest.parent/r.map_path,allow_pickle=False)) for _,r in f.iterrows()])
    y=np.array([PATTERNS.index(label) for label in f.label])
    train=f.split.eq('train').to_numpy()
    test=f.split.eq('test').to_numpy()
    model=RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=2)
    start=time.perf_counter()
    model.fit(x[train],y[train])
    pred=model.predict(x[test])
    matrix=confusion_matrix(y[test],pred,labels=np.arange(9))
    majority=int(np.bincount(y[train],minlength=9).argmax())
    report={'model':'RandomForest (200 trees, 87 spatial density features)','classes':PATTERNS,
      'accuracy':accuracy_score(y[test],pred),'macro_f1_all_9_classes':f1_score(y[test],pred,labels=np.arange(9),average='macro',zero_division=0),
      'confusion_matrix':matrix.tolist(),'support':matrix.sum(1).tolist(),
      'train_majority_baseline_accuracy':float(np.mean(y[test]==majority)),
      'fit_and_test_seconds':time.perf_counter()-start,'seed':42,'note':'Same predefined split as ViT; no test-driven tuning.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
