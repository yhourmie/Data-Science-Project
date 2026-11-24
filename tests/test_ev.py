"""Fast tests on synthetic data: no download, no MLflow, runs in CI in seconds."""

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dsp.ev.data import BEV, CATEGORICAL, FEATURES, PHEV, TARGET, clean, validate
from dsp.ev.pipeline import best_f1_threshold, build


@pytest.fixture
def synthetic() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({c: rng.choice(["a", "b", "c"], n) for c in CATEGORICAL})
    df["postal_code"] = rng.choice([str(z) for z in range(98001, 98060)], n)
    df["model_year"] = rng.integers(2012, 2027, n)
    df["longitude"] = rng.uniform(-124, -117, n)
    df["latitude"] = rng.uniform(45.6, 49, n)
    df.loc[:9, ["longitude", "latitude"]] = np.nan
    df[TARGET] = (rng.random(n) < 0.1 + 0.3 * (df["county"] == "a")).astype(int)
    return df


def raw_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "County": ["King", "Yakima", None, "Ada"],
            "City": ["Seattle", "Yakima", None, "Boise"],
            "State": ["WA", "WA", "WA", "ID"],
            "Postal Code": [98101.0, 98902.0, np.nan, 83702.0],
            "Model Year": [2023, 2017, 2020, 2022],
            "Make": ["TESLA", "TOYOTA", "KIA", "FORD"],
            "Electric Vehicle Type": [BEV, PHEV, BEV, BEV],
            "Electric Range": [0, 25, 0, 0],
            "Legislative District": [43.0, 14.0, np.nan, np.nan],
            "DOL Vehicle ID": [1, 2, 3, 4],
            "Vehicle Location": ["POINT (-122.33 47.61)", "POINT (-120.5 46.6)", None, None],
            "Electric Utility": ["CITY OF SEATTLE - (WA)", "PACIFICORP", None, None],
        }
    )


def test_clean_keeps_wa_drops_leaky_columns_and_parses_location():
    out = clean(raw_rows())
    assert len(out) == 3  # Idaho registration dropped
    assert set(out.columns) == set(FEATURES + [TARGET])  # no make, model or range
    assert out[TARGET].tolist() == [0, 1, 0]
    assert out.loc[0, "postal_code"] == "98101" and out.loc[0, "legislative_district"] == "43"
    assert out.loc[0, "city"] == "SEATTLE"
    assert out.loc[0, ["longitude", "latitude"]].tolist() == [-122.33, 47.61]
    assert out.loc[2, "county"] == "unknown" and np.isnan(out.loc[2, "longitude"])


def test_validate_rejects_bad_data(synthetic):
    validate(synthetic)
    with pytest.raises(ValueError, match="missing"):
        validate(synthetic.drop(columns=["county"]))
    bad = synthetic.copy()
    bad.loc[0, "latitude"] = 10.0
    with pytest.raises(ValueError, match="coordinates"):
        validate(bad)


@pytest.mark.parametrize(
    "name,hp",
    [
        ("logreg", {"C": 1.0}),
        ("hist_gb", {"max_iter": 20}),
        ("xgboost", {"n_estimators": 20, "max_depth": 3}),
    ],
)
def test_pipelines_fit_and_handle_unseen_category(synthetic, name, hp):
    model = build(name, hp, seed=0).fit(synthetic[FEATURES], synthetic[TARGET])
    new = synthetic[FEATURES].head(3).copy()
    new["postal_code"] = "00000"
    new.loc[new.index[0], ["longitude", "latitude"]] = np.nan
    proba = model.predict_proba(new)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_best_f1_threshold():
    y = np.array([1, 0, 0, 1])
    p = np.array([0.9, 0.1, 0.6, 0.8])
    thr, f1 = best_f1_threshold(y, p)
    assert 0.6 < thr <= 0.8 and f1 == 1.0


def test_api_predict(synthetic, tmp_path, monkeypatch):
    model = build("hist_gb", {"max_iter": 20}, seed=0).fit(synthetic[FEATURES], synthetic[TARGET])
    path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "threshold": 0.3, "features": FEATURES, "winner": "hist_gb"}, path)
    monkeypatch.setenv("MODEL_PATH", str(path))

    from dsp.ev import api

    api.bundle.cache_clear()
    client = TestClient(api.app)
    assert client.get("/health").json()["model"] == "hist_gb"

    row = {
        "county": "a",
        "city": "b",
        "postal_code": "98010",
        "legislative_district": "c",
        "electric_utility": "a",
        "model_year": 2022,
        "longitude": -122.3,
        "latitude": 47.6,
    }
    no_location = {k: v for k, v in row.items() if k not in ("longitude", "latitude")}
    r = client.post("/predict", json=[row, no_location])
    assert r.status_code == 200
    for pred in r.json():
        assert 0 <= pred["phev_probability"] <= 1
        assert pred["predicted_type"] in ("PHEV", "BEV")

    assert client.post("/predict", json=[{**row, "latitude": 10.0}]).status_code == 422
