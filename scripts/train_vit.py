"""Train from a CSV manifest: map_path,label,lot_id. Splits are disjoint by lot."""
import argparse
import json
import hashlib
import platform
import time
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


def evaluate(model, loader, device='cpu'):
    matrix = np.zeros((9,9),dtype=int)
    model.eval()
    with torch.inference_mode():
        for x,y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            np.add.at(matrix,(y.numpy(),pred.numpy()),1)
    den = matrix.sum(0)+matrix.sum(1)
    f1 = np.divide(2*matrix.diagonal(),den,out=np.zeros(9),where=den>0)
    return {'accuracy':float(matrix.trace()/matrix.sum()),'macro_f1_all_9_classes':float(f1.mean()),
            'support':matrix.sum(1).tolist(),'confusion_matrix':matrix.tolist()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--output',type=Path,default=Path('checkpoints'))
    a=p.parse_args()
    if a.epochs < 1 or a.batch_size < 1:
        p.error('epochs and batch-size must be positive')
    torch.manual_seed(42)
    frame=pd.read_csv(a.manifest,dtype=str)
    if not {'map_path','label','lot_id'}.issubset(frame.columns) or frame.isna().any().any():
        p.error('Manifest requires non-null map_path,label,lot_id')
    if not frame.label.isin(PATTERNS).all() or frame.map_path.duplicated().any():
        p.error('Unknown labels or duplicated map paths')
    device='cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device)
    if device=='cuda' and not torch.cuda.is_available(): p.error('CUDA requested but unavailable in this PyTorch environment')
    if 'split' in frame:
        if not frame.split.isin(['train','validation','test']).all():p.error('Unknown split')
        splits=[frame[frame.split==s].copy() for s in ['train','validation','test']]
        if any(f.empty for f in splits):p.error('Each split must contain maps')
        lots=[set(f.lot_id) for f in splits]
        if any(lots[i]&lots[j] for i in range(3) for j in range(i)):p.error('Lot leakage across predefined splits')
        if 'map_sha256' in frame and frame.map_sha256.duplicated().any():p.error('Duplicate map content in manifest')
    else:
        splits=split_lots(frame)
    loaders=[DataLoader(Maps(f,a.manifest.parent),batch_size=a.batch_size,shuffle=i==0) for i,f in enumerate(splits)]
    model=WaferViT().to(device)
    print('Training device:',device,flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    a.output.mkdir(parents=True,exist_ok=True)
    best=-1
    history=[]
    start=time.perf_counter()
    for epoch in range(a.epochs):
        model.train()
        for x,y in loaders[0]:
            x,y=x.to(device),y.to(device)
            opt.zero_grad()
            loss=torch.nn.functional.cross_entropy(model(x),y)
            loss.backward()
            opt.step()
        metrics=evaluate(model,loaders[1],device)
        history.append({'epoch':epoch+1,**metrics})
        print({'epoch':epoch+1,'accuracy':metrics['accuracy'],'macro_f1':metrics['macro_f1_all_9_classes']},flush=True)
        if metrics['macro_f1_all_9_classes'] > best:
            best=metrics['macro_f1_all_9_classes']
            torch.save(model.state_dict(),a.output/'vit.pt')
    model.load_state_dict(torch.load(a.output/'vit.pt',weights_only=True,map_location=device))
    test_metrics=evaluate(model,loaders[2],device)
    if device=='cuda':torch.cuda.synchronize()
    report={'classes':PATTERNS,'seed':42,'history':history,'test':test_metrics,
            'device':device,'device_name':torch.cuda.get_device_name(0) if device=='cuda' else 'CPU',
            'torch_version':torch.__version__,'cuda_runtime':torch.version.cuda,
            'batch_size':a.batch_size,'epochs':a.epochs,
            'python':platform.python_version(),'train_evaluate_seconds':time.perf_counter()-start,
            'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
            'checkpoint_sha256':hashlib.sha256((a.output/'vit.pt').read_bytes()).hexdigest(),
            'lots':{k:sorted(f.lot_id.unique().tolist()) for k,f in zip(['train','validation','test'],splits)},
            'note':'Research run; metrics apply only to this manifest. Missing-class F1=0.'}
    (a.output/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
