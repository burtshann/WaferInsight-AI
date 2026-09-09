import pytest
torch=pytest.importorskip('torch')
from waferinsight.improved import SpatialInspector,categorical_batch,classification_metrics


@pytest.mark.parametrize('architecture',['cnn','hybrid'])
def test_spatial_encoder_gradients(architecture):
    net=SpatialInspector(architecture)
    maps=torch.randint(0,3,(2,64,64),dtype=torch.uint8)
    out=net(categorical_batch(maps))
    assert out.shape==(2,9) and torch.isfinite(out).all()
    torch.nn.functional.cross_entropy(out,torch.tensor([1,8])).backward()
    assert net.stem[0].weight.grad.abs().sum()>0


def test_categorical_augmentation_preserves_die_counts():
    maps=torch.randint(0,3,(4,64,64),dtype=torch.uint8)
    x=categorical_batch(maps,augment=True)
    assert torch.equal(x.sum(1),torch.ones(4,64,64))
    assert torch.equal(x.sum((-1,-2)),categorical_batch(maps).sum((-1,-2)))


def test_missing_class_metrics_are_explicit():
    m=classification_metrics([0,0,1],[0,1,1])
    assert m['support']==[2,1,0,0,0,0,0,0,0]
    assert m['accuracy']==pytest.approx(2/3)
    assert m['macro_f1_supported_classes']==pytest.approx(2/3)
    assert m['macro_f1_all_9_classes']==pytest.approx(4/27)


def test_inference_rejects_tampered_checkpoint(tmp_path):
    import json
    from waferinsight.core import PATTERNS
    from waferinsight.predict import load_inspector
    (tmp_path/'model_card.json').write_text(json.dumps({'classes':PATTERNS,'checkpoint_sha256':'wrong','architecture':'cnn'}))
    (tmp_path/'wafer_inspector.pt').write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError,match='checksum'):load_inspector(tmp_path)
