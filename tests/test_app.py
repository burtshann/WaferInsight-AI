from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_demo_and_pattern_switch():
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=30)
    assert not app.exception
    assert len(app.metric)==4
    app.selectbox[0].select('Scratch').run()
    assert not app.exception


def test_trained_classifier_ui():
    import pytest
    pytest.importorskip('torch')
    if not (Path(__file__).resolve().parents[1]/'models'/'model_card.json').exists():
        pytest.skip('Packaged model is not available')
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=30)
    app.checkbox[0].check().run(timeout=30)
    assert not app.exception
    assert any('Predicted pattern:' in element.value for element in app.markdown)
