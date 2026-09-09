"""Local spatial encoders for categorical wafer maps; no pretrained weights."""
import torch
from torch import nn
import torch.nn.functional as F


class SpatialInspector(nn.Module):
    def __init__(self, architecture='cnn'):
        super().__init__()
        if architecture not in ['cnn','hybrid']:raise ValueError('Unknown architecture')
        self.architecture=architecture
        layers=[]
        channels=3
        for width in [32,64,96]:
            layers.extend([nn.Conv2d(channels,width,3,stride=2,padding=1,bias=False),
                nn.BatchNorm2d(width),nn.GELU(),nn.Conv2d(width,width,3,padding=1,bias=False),
                nn.BatchNorm2d(width),nn.GELU()])
            channels=width
        self.stem=nn.Sequential(*layers)
        if architecture=='hybrid':
            self.position=nn.Parameter(torch.randn(1,64,96)*.02)
            self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(96,4,192,dropout=.1,batch_first=True),2)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(96*4*4,192),nn.GELU(),nn.Dropout(.2),nn.Linear(192,9))

    def forward(self,x):
        x=self.stem(x)
        if self.architecture=='hybrid':
            x=self.encoder(x.flatten(2).transpose(1,2)+self.position).transpose(1,2).reshape(-1,96,8,8)
        return self.head(F.adaptive_avg_pool2d(x,(4,4)))


def categorical_batch(maps,augment=False):
    x=F.one_hot(maps.long(),num_classes=3).permute(0,3,1,2).float()
    if augment:
        x=torch.rot90(x,int(torch.randint(4,()).item()),(-2,-1))
        if torch.rand(()).item()<.5:x=x.flip(-1)
    return x


def classification_metrics(labels,predictions):
    matrix=torch.zeros(9,9,dtype=torch.int64)
    for y,p in zip(labels,predictions):matrix[int(y),int(p)]+=1
    m=matrix.float()
    support=m.sum(1)
    precision=m.diag()/m.sum(0).clamp_min(1)
    recall=m.diag()/support.clamp_min(1)
    f1=2*m.diag()/(m.sum(0)+support).clamp_min(1)
    return {'accuracy':float(m.diag().sum()/m.sum()),'macro_f1_all_9_classes':float(f1.mean()),
        'macro_f1_supported_classes':float(f1[support>0].mean()),'precision':precision.tolist(),
        'recall':recall.tolist(),'f1':f1.tolist(),'support':support.int().tolist(),'confusion_matrix':matrix.tolist()}
