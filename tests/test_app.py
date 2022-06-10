from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_renders_recommendation():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert len(at.success) + len(at.warning) + len(at.error) >= 1
    assert any("Effect on order value" in m.label for m in at.metric)
