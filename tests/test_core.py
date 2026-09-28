import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from heatwave import INPUT_COLUMNS
from heatwave.calendar import doy365
from heatwave.climatology import compute_normals
from heatwave.labeling import label_observations, BASE_THRESHOLDS
from heatwave.pipeline import FeatureBuilder, make_pipeline
from heatwave.model import save_artifact, load_artifact, neighbor_breakdown, split_by_year
from heatwave.demo import predict_observation


def frame(t=43., n=37., region="plains"):
    return pd.DataFrame({"date": ["2024-03-01"], "lat": [19.5], "lon": [75.5], "region_type": [region], "max_temp": [t], "normal_temp": [n]})


@pytest.mark.parametrize("region", BASE_THRESHOLDS)
@pytest.mark.parametrize("departure,expected", [(4.4, 0), (4.5, 1), (6.4, 1), (6.5, 2)])
def test_departure_boundaries(region, departure, expected):
    base = BASE_THRESHOLDS[region]
    assert label_observations(frame(base, base-departure, region)).iloc[0] == expected
    assert label_observations(frame(base-.01, base-.01-departure, region)).iloc[0] == 0


@pytest.mark.parametrize("region", BASE_THRESHOLDS)
@pytest.mark.parametrize("temperature,expected", [(44.9, 0), (45, 1), (46.9, 1), (47, 2)])
def test_absolute_boundaries(region, temperature, expected):
    assert label_observations(frame(temperature, 43, region)).iloc[0] == expected


def test_absolute_normal_gate_and_higher_rule():
    assert label_observations(frame(45, 40)).iloc[0] == 1
    assert label_observations(frame(44.9, 38.4)).iloc[0] == 2
    assert label_observations(frame(47, 41)).iloc[0] == 2


def test_calendar():
    data = pd.concat([frame(), frame()], ignore_index=True)
    data["date"] = ["2023-03-01", "2024-03-01"]
    built = FeatureBuilder().fit_transform(data)
    assert built.doy365.tolist() == [60, 60]
    np.testing.assert_allclose(built.iloc[0][["doy_sin", "doy_cos"]].astype(float), built.iloc[1][["doy_sin", "doy_cos"]].astype(float))
    with pytest.raises(ValueError, match="February 29"):
        doy365(["2024-02-29"])


def test_pooled_normal_excludes_leap_day_and_future():
    records = []
    for year in range(1991, 2015):
        for day in pd.date_range(f"{year}-02-22", f"{year}-03-08"):
            records.append({"date": day, "cell_id": "a", "max_temp": 999 if day.month == 2 and day.day == 29 else 10.})
    baseline = pd.DataFrame(records)
    baseline.loc[0, "max_temp"] = np.nan
    baseline.loc[1, "max_temp"] = 30
    original = compute_normals(baseline)
    extra = pd.DataFrame({"date": [pd.Timestamp("2024-03-01")], "cell_id": ["a"], "max_temp": [9999.]})
    actual = compute_normals(pd.concat([baseline, extra], ignore_index=True))
    pd.testing.assert_frame_equal(original, actual)
    march = actual[actual.doy365 == 60].iloc[0]
    assert march.normal_count == 359
    assert march.normal_temp == pytest.approx((358*10+30)/359)


@pytest.fixture
def training():
    data = pd.concat([frame(t, 34 + i % 4, list(BASE_THRESHOLDS)[i % 3]) for i, t in enumerate(np.linspace(29, 49, 80))], ignore_index=True)
    data["date"] = [f"{2015+i%8}-04-01" for i in range(len(data))]
    data["label"] = label_observations(data)
    data["departure"] = data.max_temp-data.normal_temp
    return data


def test_pipeline_scaler_and_departure(training):
    model = make_pipeline().fit(training[INPUT_COLUMNS], training.label)
    built = model.named_steps["features"].transform(training)
    np.testing.assert_allclose(built.departure, training.departure)
    scaler = model.named_steps["preprocessing"].named_transformers_["numeric"]
    assert scaler.mean_[0] == pytest.approx(training.max_temp.mean())
    assert scaler.n_samples_seen_ == len(training)
    before = scaler.mean_.copy()
    model.predict(frame(60, 30))
    np.testing.assert_array_equal(before, scaler.mean_)
    with pytest.raises(ValueError, match="Unknown"):
        model.predict(frame(region="unknown"))


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_artifact_subprocess_app_and_votes(training, tmp_path, weights):
    model = make_pipeline(k=5, weights=weights).fit(training[INPUT_COLUMNS], training.label)
    path = tmp_path / "artifact.joblib"
    save_artifact(model, path)
    raw = training.iloc[[0]][INPUT_COLUMNS].copy()
    raw.to_json(tmp_path / "input.json", orient="table", date_format="iso")
    code = "from heatwave.model import load_artifact; import pandas as pd, sys; a=load_artifact(sys.argv[1]); x=pd.read_json(sys.argv[2],orient='table'); print(int(a['pipeline'].predict(x)[0]))"
    result = subprocess.check_output([sys.executable, "-c", code, str(path), str(tmp_path / "input.json")], text=True)
    assert int(result.strip()) == model.predict(raw)[0]
    loaded = load_artifact(path)
    cells = pd.DataFrame({"cell_id": ["19.5_75.5"], "lat": [19.5], "lon": [75.5], "region_type": [raw.region_type.iloc[0]]})
    output = predict_observation(loaded, cells, **raw.iloc[0].drop("region_type").to_dict())
    assert output["prediction"] == loaded["label_map"][int(model.predict(raw)[0])]
    breakdown, _ = neighbor_breakdown(model, raw)
    expected = pd.Series(model.predict_proba(raw)[0], index=[loaded["label_map"][int(c)] for c in model.classes_])
    np.testing.assert_allclose(breakdown.vote_share.loc[expected.index], expected)
    assert breakdown.neighbors.sum() == 5


def test_year_split(training):
    test = frame()
    test["label"] = 1
    train, held = split_by_year(pd.concat([training, test], ignore_index=True))
    assert pd.to_datetime(train.date).dt.year.max() == 2022
    assert pd.to_datetime(held.date).dt.year.min() == 2024


def test_scaler_is_refit_per_year_fold(training):
    from sklearn.model_selection import LeaveOneGroupOut, cross_validate
    years = pd.to_datetime(training.date).dt.year
    result = cross_validate(make_pipeline(), training[INPUT_COLUMNS], training.label, groups=years, cv=LeaveOneGroupOut(), return_estimator=True)
    for (_, valid), estimator in zip(LeaveOneGroupOut().split(training, groups=years), result['estimator']):
        held_year = years.iloc[valid].iloc[0]
        expected = training.loc[years != held_year, 'max_temp'].mean()
        assert estimator.named_steps['preprocessing'].named_transformers_['numeric'].mean_[0] == pytest.approx(expected)


def test_grd_orientation_missing_and_size(tmp_path):
    from heatwave.data_loader import read_grd
    array = np.full((366, 31, 31), 30, dtype='<f4')
    array[60, 12, 8] = 42
    array[61, 12, 8] = 99.9
    path = tmp_path / '2024.GRD'
    array.tofile(path)
    cells = pd.DataFrame({'cell_id':['19.5_75.5'], 'lat':[19.5], 'lon':[75.5], 'i':[12], 'j':[8]})
    result = read_grd(path, 2024, cells)
    assert result.loc[60, 'date'] == pd.Timestamp('2024-03-01')
    assert result.loc[60, 'max_temp'] == 42
    assert pd.isna(result.loc[61, 'max_temp'])
    with pytest.raises(ValueError, match='expected'):
        read_grd(path, 2023, cells)
    with pytest.raises(ValueError, match='Implausible'):
        read_grd(path, 2024, cells, '>')


def test_artifact_version_rejected(training, tmp_path):
    import joblib
    model = make_pipeline().fit(training[INPUT_COLUMNS], training.label)
    path = tmp_path / 'model.joblib'
    obj = save_artifact(model, path)
    obj['sklearn_version'] = '0.0.0'
    joblib.dump(obj, path)
    with pytest.raises(ValueError, match='scikit-learn'):
        load_artifact(path)


def test_noise_only_changes_maximum(training):
    from heatwave.evaluate import noise_robustness
    model = make_pipeline().fit(training[INPUT_COLUMNS], training.label)
    original = training.copy(deep=True)
    result = noise_robustness(model, training, training, ks=[1], sigmas=[.5], seeds=[11])
    assert set(result.comparison) == {'Original labels', 'Recomputed rules', 'Rules vs original'}
    pd.testing.assert_frame_equal(training, original)


def test_app_initial_render():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
    assert not app.exception
