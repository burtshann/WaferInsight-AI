"""Verified local model loading and wafer-map pattern inference."""
import hashlib
import json
from pathlib import Path
import torch
from waferinsight.core import PATTERNS,validate_map
from waferinsight.model import preprocess
from waferinsight.improved import SpatialInspector


DEFAULT_MODEL=Path(__file__).resolve().parents[1]/'models'


def load_inspector(directory=DEFAULT_MODEL):
    directory=Path(directory)
    card=json.loads((directory/'model_card.json').read_text(encoding='utf-8'))
    checkpoint=directory/'wafer_inspector.pt'
    if card['classes']!=PATTERNS:raise ValueError('Model class mapping mismatch')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=card['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    model=SpatialInspector(card['architecture'])
    model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'))
    model.eval()
    return model,card


def inspect_map(a,model):
    a=validate_map(a)
    with torch.inference_mode():
        scores=model(preprocess(a)[None]).softmax(-1)[0].tolist()
    index=max(range(len(scores)),key=scores.__getitem__)
    return {'pattern':PATTERNS[index],'score':scores[index],
            'class_scores':dict(zip(PATTERNS,scores)),
            'note':'Uncalibrated model scores; wafer-map pattern classification, not AOI defect detection or a production release decision.'}
