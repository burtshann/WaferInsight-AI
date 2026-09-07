"""Train from a CSV manifest: map_path,label,lot_id. Splits are disjoint by lot."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from waferinsight.core import PATTERNS, validate_map
from waferinsight.model import WaferViT, preprocess


def split_lots(frame, seed=42):
    lots = np.array(sorted(frame.lot_id.unique()))
    if len(lots) < 3:
        raise ValueError('Need at least 3 distinct lots for train/validation/test')
    np.random.default_rng(seed).shuffle(lots)
    n = max(1,int(len(lots)*.15))
    return [frame[frame.lot_id.isin(g)].copy() for g in [lots[2*n:],lots[:n],lots[n:2*n]]]


class Maps(Dataset):
    def __init__(self, frame, root):
        self.frame, self.root = frame.reset_index(drop=True),root
    def __len__(self):
        return len(self.frame)
    def __getitem__(self,i):
        r = self.frame.iloc[i]
        a = validate_map(np.load(self.root/r.map_path,allow_pickle=False))
        return preprocess(a), PATTERNS.index(r.label)


def evaluate(model, loader):
    matrix = np.zeros((9,9),dtype=int)
    model.eval()
    with torch.inference_mode():
        for x,y in loader:
            pred = model(x).argmax(1)
            np.add.at(matrix,(y.numpy(),pred.numpy()),1)
    den = matrix.sum(0)+matrix.sum(1)
    f1 = np.divide(2*matrix.diagonal(),den,out=np.zeros(9),where=den>0)
    return {'accuracy':float(matrix.trace()/matrix.sum()),'macro_f1_all_9_classes':float(f1.mean()),
            'support':matrix.sum(1).tolist(),'confusion_matrix':matrix.tolist()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--output',type=Path,default=Path('checkpoints'))
    a=p.parse_args()
    if a.epochs < 1:
        p.error('epochs must be positive')
    torch.manual_seed(42)
    frame=pd.read_csv(a.manifest,dtype=str)
    if not {'map_path','label','lot_id'}.issubset(frame.columns) or frame.isna().any().any():
        p.error('Manifest requires non-null map_path,label,lot_id')
    if not frame.label.isin(PATTERNS).all() or frame.map_path.duplicated().any():
        p.error('Unknown labels or duplicated map paths')
    splits=split_lots(frame)
    loaders=[DataLoader(Maps(f,a.manifest.parent),batch_size=32,shuffle=i==0) for i,f in enumerate(splits)]
    model=WaferViT()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    a.output.mkdir(parents=True,exist_ok=True)
    best=-1
    history=[]
    for epoch in range(a.epochs):
        model.train()
        for x,y in loaders[0]:
            opt.zero_grad()
            loss=torch.nn.functional.cross_entropy(model(x),y)
            loss.backward()
            opt.step()
        metrics=evaluate(model,loaders[1])
        history.append({'epoch':epoch+1,**metrics})
        print(history[-1])
        if metrics['macro_f1_all_9_classes'] > best:
            best=metrics['macro_f1_all_9_classes']
            torch.save(model.state_dict(),a.output/'vit.pt')
    model.load_state_dict(torch.load(a.output/'vit.pt',weights_only=True))
    report={'classes':PATTERNS,'seed':42,'history':history,'test':evaluate(model,loaders[2]),
            'lots':{k:sorted(f.lot_id.unique().tolist()) for k,f in zip(['train','validation','test'],splits)},
            'note':'Research run; metrics apply only to this manifest. Missing-class F1=0. CPU training.'}
    (a.output/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
