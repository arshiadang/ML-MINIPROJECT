"""Exercise state changes and the saved-evidence pages in the real demo."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / 'models/knn_heatwave.joblib').exists(), reason='Requires delivered artifact')


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_prediction_refresh_and_severe_comparison():
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
    button(app, 'Classify observation →').click().run(timeout=30)
    assert not app.exception
    assert any(m.label == 'Project rule label' and m.value == 'Heatwave' for m in app.metric)
    app.number_input(key='tmax').set_value(36.).run(timeout=30)
    assert not any(m.label == 'Project rule label' for m in app.metric)
    assert any('Your inputs changed' in m.value for m in app.markdown)
    button(app, 'Severe rule case').click().run(timeout=30)
    button(app, 'Classify observation →').click().run(timeout=30)
    assert not app.exception
    assert any(m.label == 'Project rule label' and m.value == 'Severe Heatwave' for m in app.metric)
    assert any('Model and rule disagree' in w.value for w in app.warning)


def test_pages_and_out_of_region_validation():
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
    for label in ['Model evidence','Region map','Method & data','Classify a day']:
        app.radio[0].set_value(label).run(timeout=30)
        assert not app.exception, label
    latitude = next(n for n in app.number_input if n.label == 'Latitude (°N)')
    latitude.set_value(15.).run(timeout=30)
    button(app,'Classify observation →').click().run(timeout=30)
    assert not app.exception
    assert any('outside Maharashtra' in e.value for e in app.error)
    assert not any(m.label == 'Project rule label' for m in app.metric)
