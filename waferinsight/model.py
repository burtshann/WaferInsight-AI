"""Small ViT trained from scratch; no pretrained weights or accuracy claim."""
import torch
from torch import nn
import torch.nn.functional as F


def preprocess(a):
    t = torch.tensor(a.copy(), dtype=torch.long)
    channels = F.one_hot(t, num_classes=3).permute(2,0,1).float()[None]
    return F.interpolate(channels, size=(64,64), mode='nearest')[0]


class WaferViT(nn.Module):
    def __init__(self, classes=9):
        super().__init__()
        self.patch = nn.Conv2d(3, 64, 8, stride=8)
        self.position = nn.Parameter(torch.randn(1,64,64)*.02)
        layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64,classes))

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1,2) + self.position
        return self.head(self.encoder(x).mean(1))
