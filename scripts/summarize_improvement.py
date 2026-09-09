"""Package validation-selected weights and render the completed evaluation."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from waferinsight.core import PATTERNS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reports',type=Path,default=Path('reports/improvement'))
    parser.add_argument('--checkpoints',type=Path,default=Path('checkpoints/improvement'))
    parser.add_argument('--model-dir',type=Path,default=Path('models'))
    args=parser.parse_args()
    root=args.reports
    result=json.loads((root/'final_evaluation.json').read_text())
    selection=result['selection']
    directory=args.model_dir;directory.mkdir(parents=True,exist_ok=True)
    source=args.checkpoints/f"{selection['architecture']}.pt"
    if hashlib.sha256(source.read_bytes()).hexdigest()!=selection['checkpoint_sha256']:raise ValueError('Checkpoint mismatch')
    shutil.copyfile(source,directory/'wafer_inspector.pt')
    card={**selection,'classes':PATTERNS,'input':'2D categorical wafer map: 0 outside,1 pass,2 fail; nearest-neighbour 64x64 one-hot encoding',
      'training_dataset':'WM-811K; 13,933 maps from original training lots','weights':'state_dict; load with weights_only=True',
      'limitations':'Research classification only. Uncalibrated scores. Fresh audit has no Donut or Near-full. Not production-qualified.',
      'license':'MIT for model/code; source dataset listing CC0; see report for provenance',
      'report':os.path.relpath(root/'README.md',directory).replace('\\','/')}
    (directory/'model_card.json').write_text(json.dumps(card,indent=2),encoding='utf-8')
    manifest=pd.read_csv(root/'split_manifest.csv')
    predictions=pd.read_csv(root/'predictions.csv')
    fresh=predictions[predictions.split=='fresh_audit'].copy()
    fresh['correct']=fresh.true_label.eq(fresh.predicted_label).astype(int)
    paired=fresh.pivot(index='source_index',columns='model',values='correct').merge(manifest[['source_index','lot_id']],on='source_index')
    by_lot=paired.groupby('lot_id').agg(n=('selected','size'),new=('selected','sum'),old=('original_vit','sum'))
    values=by_lot.to_numpy();rng=np.random.default_rng(42);differences=[]
    for _ in range(2000):
        b=values[rng.integers(len(values),size=len(values))].sum(0)
        differences.append((b[1]-b[2])/b[0])
    uncertainty={'method':'paired lot-cluster percentile bootstrap,2000 resamples,seed42',
      'lots':len(by_lot),'accuracy_gain':float((paired.selected-paired.original_vit).mean()),
      'accuracy_gain_95pct_interval':np.quantile(differences,[.025,.975]).tolist()}
    (root/'uncertainty.json').write_text(json.dumps(uncertainty,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for name in ['cnn','hybrid']:
        r=json.loads((root/f'{name}_validation.json').read_text())
        axes[0].plot([h['epoch'] for h in r['history']],[h['validation']['macro_f1_all_9_classes'] for h in r['history']],label=name)
    axes[0].axhline(.7034201622832099,color='gray',linestyle='--',label='original ViT best validation')
    axes[0].set(xlabel='Epoch',ylabel='Validation macro-F1',title='Validation-only model selection',ylim=(.5,1));axes[0].legend()
    values=[];names=[]
    for split in ['historical_test','fresh_audit']:
        for model in ['original_vit','selected']:
            names.append(f"{split.replace('_',' ')}\n{model.replace('_',' ')}")
            values.append(result['scores'][split][model]['accuracy'])
    axes[1].bar(names,values,color=['#8c99a8','#1c9c89']*2)
    axes[1].set(ylabel='Accuracy',ylim=(0,1),title='Historical comparison + new seven-class audit')
    axes[1].tick_params(axis='x',labelsize=8)
    for i,v in enumerate(values):axes[1].text(i,v+.015,f'{v:.1%}',ha='center')
    fig.tight_layout();fig.savefig(root/'improvement.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for ax,split in zip(axes,['historical_test','fresh_audit']):
        m=np.array(result['scores'][split]['selected']['confusion_matrix'])
        ax.imshow(m/np.maximum(1,m.sum(1,keepdims=True)),vmin=0,vmax=1,cmap='Blues')
        for y in range(9):
            for x in range(9):ax.text(x,y,str(m[y,x]),ha='center',va='center',fontsize=8,color='white' if m[y,x]/max(1,m[y].sum())>.5 else 'black')
        ax.set_xticks(range(9),PATTERNS,rotation=50,ha='right');ax.set_yticks(range(9),PATTERNS)
        ax.set(xlabel='Predicted',ylabel='True',title=split.replace('_',' ')+' / selected model')
    fig.tight_layout();fig.savefig(root/'confusion_matrices.png',dpi=150);plt.close(fig)
    print(json.dumps({'selection':selection,'uncertainty':uncertainty},indent=2))


if __name__=='__main__':main()
