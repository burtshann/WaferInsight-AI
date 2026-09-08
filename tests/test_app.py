from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_demo_and_pattern_switch():
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=30)
    assert not app.exception
    assert len(app.metric)==4
    app.selectbox[0].select('Scratch').run()
    assert not app.exception
