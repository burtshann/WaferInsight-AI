"""Render actual public examples and measured classifier results."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from waferinsight.core import PATTERNS


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--report',type=Path,default=Path('reports/wm811k'))
    a=p.parse_args()
    f=pd.read_csv(a.manifest,dtype=str)
    fig,axes=plt.subplots(3,3,figsize=(11,10),facecolor='#0b1220')
    cmap=ListedColormap(['#0b1220','#25b7a1','#ff635e'])
    for ax,label in zip(axes.flat,PATTERNS):
        r=f[(f.label==label)&(f.split=='test')].iloc[0]
        wafer=np.load(a.manifest.parent/r.map_path,allow_pickle=False)
        ax.imshow(wafer,cmap=cmap,vmin=0,vmax=2,interpolation='nearest')
        ax.set_title(f'{label}  |  source row {r.source_index}',color='white',fontsize=11)
        ax.axis('off')
    fig.suptitle('WM-811K  /  REAL HELD-OUT WAFER MAPS',color='white',fontsize=20)
    fig.tight_layout(rect=[0,0,1,.96])
    fig.savefig(a.report/'public_examples.png',dpi=140,facecolor=fig.get_facecolor())
    plt.close(fig)
    vit_path=a.report/'vit_evaluation.json'
    if vit_path.exists():
        vit=json.loads(vit_path.read_text())['test']
        baseline=json.loads((a.report/'baseline.json').read_text())
        fig,axes=plt.subplots(1,2,figsize=(15,6))
        for ax,title,result in zip(axes,['Small ViT','Spatial Random Forest'],[vit,baseline]):
            m=np.array(result['confusion_matrix'])
            ax.imshow(m/np.maximum(1,m.sum(1,keepdims=True)),vmin=0,vmax=1,cmap='Blues')
            for y in range(9):
                for x in range(9):
                    ax.text(x,y,str(m[y,x]),ha='center',va='center',fontsize=8,color='white' if m[y,x]/max(1,m[y].sum())>.5 else 'black')
            ax.set_xticks(range(9),PATTERNS,rotation=55,ha='right')
            ax.set_yticks(range(9),PATTERNS)
            ax.set_xlabel('Predicted');ax.set_ylabel('True')
            ax.set_title(f"{title}\nAccuracy {result['accuracy']:.1%} | macro-F1 {result['macro_f1_all_9_classes']:.3f}")
        fig.suptitle('WM-811K public subset / identical lot-disjoint test split')
        fig.tight_layout()
        fig.savefig(a.report/'confusion_matrices.png',dpi=140)
        plt.close(fig)
