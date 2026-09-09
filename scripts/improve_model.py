"""Validation-only candidate training and separate, explicitly sealed final evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler
from waferinsight.core import PATTERNS
from waferinsight.improved import SpatialInspector,categorical_batch,classification_metrics
from waferinsight.model import WaferViT


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(root,device):
    f=pd.read_csv(root/'manifest.csv')
    if not f.map_sha256.is_unique or f.groupby('lot_id').split.nunique().max()!=1:
        raise ValueError('Duplicate maps or lot leakage')
    maps=torch.from_numpy(np.load(root/'maps.npy',allow_pickle=False)).to(device)
    labels=torch.tensor([PATTERNS.index(x) for x in f.label],device=device)
    return f,maps,labels


def predict(model,maps,labels,index,batch=128):
    model.eval()
    predictions=[]
    with torch.inference_mode():
        for ids in index.split(batch):
            predictions.extend(model(categorical_batch(maps[ids])).argmax(1).cpu().tolist())
    truth=labels[index].cpu().tolist()
    return classification_metrics(truth,predictions),predictions


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['train','evaluate'])
    p.add_argument('--data',type=Path,default=Path('data/improvement'))
    p.add_argument('--output',type=Path,default=Path('checkpoints/improvement'))
    p.add_argument('--reports',type=Path,default=Path('reports/improvement'))
    p.add_argument('--architecture',choices=['cnn','hybrid'],default='cnn')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--original-checkpoint',type=Path,default=Path('checkpoints/wm811k/vit.pt'))
    a=p.parse_args()
    if not torch.cuda.is_available():p.error('This recorded experiment requires CUDA')
    if a.epochs<1 or a.batch_size<1:p.error('Positive epochs and batch size required')
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device='cuda'
    f,maps,labels=load_data(a.data,device)
    indices={s:torch.tensor(g.index.to_numpy(),device=device) for s,g in f.groupby('split')}
    a.output.mkdir(parents=True,exist_ok=True)
    a.reports.mkdir(parents=True,exist_ok=True)
    environment={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
      'gpu':torch.cuda.get_device_name(0),'manifest_sha256':sha(a.data/'manifest.csv'),'seed':42}
    if a.action=='train':
        report_path=a.reports/f'{a.architecture}_validation.json'
        if report_path.exists():p.error('Candidate report already exists; preserve it and choose a new output experiment')
        model=SpatialInspector(a.architecture).to(device)
        train=indices['train']; valid=indices['validation']
        train_labels=labels[train].cpu()
        counts=torch.bincount(train_labels,minlength=9)
        if (counts==0).any():p.error('Training class absent')
        sampler=WeightedRandomSampler((1/counts.float())[train_labels],len(train),replacement=True)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=1e-5)
        history=[];best=-1;best_epoch=0
        torch.cuda.synchronize();start=time.perf_counter()
        for epoch in range(a.epochs):
            model.train()
            order=train[torch.tensor(list(sampler),device=device)]
            losses=[]
            for ids in order.split(a.batch_size):
                opt.zero_grad(set_to_none=True)
                logits=model(categorical_batch(maps[ids],augment=True))
                loss=torch.nn.functional.cross_entropy(logits,labels[ids],label_smoothing=.05)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step();losses.append(loss.detach())
            metrics,_=predict(model,maps,labels,valid,a.batch_size)
            history.append({'epoch':epoch+1,'train_loss':float(torch.stack(losses).mean()),'validation':metrics})
            score=metrics['macro_f1_all_9_classes']
            if score>best:
                best=score;best_epoch=epoch+1
                torch.save(model.state_dict(),a.output/f'{a.architecture}.pt')
            scheduler.step()
            print(a.architecture,epoch+1,'val_accuracy',round(metrics['accuracy'],4),'val_macro_f1',round(score,4),flush=True)
        torch.cuda.synchronize()
        report={**environment,'architecture':a.architecture,'epochs':a.epochs,'batch_size':a.batch_size,
          'best_epoch':best_epoch,'best_validation_macro_f1':best,'history':history,
          'seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters()),
          'checkpoint_sha256':sha(a.output/f'{a.architecture}.pt'),'test_evaluated':False}
        report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    else:
        final=a.reports/'final_evaluation.json'
        if final.exists():p.error('Final audit already evaluated; do not tune or overwrite it')
        candidates=[json.loads((a.reports/f'{name}_validation.json').read_text()) for name in ['cnn','hybrid']]
        if any(c['manifest_sha256']!=environment['manifest_sha256'] for c in candidates):p.error('Candidate manifest mismatch')
        chosen=max(candidates,key=lambda r:r['best_validation_macro_f1'])
        architecture=chosen['architecture']
        checkpoint=a.output/f'{architecture}.pt'
        if sha(checkpoint)!=chosen['checkpoint_sha256']:p.error('Checkpoint changed after validation')
        selection={'architecture':architecture,'checkpoint_sha256':sha(checkpoint),
          'validation_macro_f1':chosen['best_validation_macro_f1'],'selection_criterion':'validation macro-F1 only'}
        (a.reports/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
        model=SpatialInspector(architecture).to(device)
        model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location=device))
        original=WaferViT().to(device)
        original.load_state_dict(torch.load(a.original_checkpoint,weights_only=True,map_location=device))
        result={**environment,'selection':selection,'classes':PATTERNS,'scores':{},
          'original_checkpoint_sha256':sha(a.original_checkpoint)}
        rows=[]
        for split in ['fresh_audit','historical_test']:
            result['scores'][split]={}
            for name,net in [('selected',model),('original_vit',original)]:
                metrics,preds=predict(net,maps,labels,indices[split])
                result['scores'][split][name]=metrics
                for idx,pred in zip(indices[split].cpu().tolist(),preds):
                    rows.append({'source_index':int(f.loc[idx,'source_index']),'split':split,'model':name,
                                 'true_label':f.loc[idx,'label'],'predicted_label':PATTERNS[pred]})
        pd.DataFrame(rows).to_csv(a.reports/'predictions.csv',index=False,lineterminator='\n')
        final.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({s:{m:{k:v[k] for k in ['accuracy','macro_f1_supported_classes']} for m,v in r.items()} for s,r in result['scores'].items()},indent=2))


if __name__=='__main__':main()
