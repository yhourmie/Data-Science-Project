"""Fast tests on synthetic data: no download, no MLflow, runs in CI in seconds."""

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dsp.bank.data import CATEGORICAL, FEATURES, NUMERIC, TARGET, clean, validate
from dsp.bank.pipeline import best_threshold, build, expected_profit, lift_at


@pytest.fixture
def synthetic() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({c: rng.choice(["a", "b", "c"], n) for c in CATEGORICAL})
    for c in NUMERIC:
        df[c] = rng.normal(size=n)
    df["age"] = rng.integers(18, 90, n)
    df[TARGET] = (rng.random(n) < 0.2 + 0.3 * (df["job"] == "a")).astype(int)
    return df


def test_clean_renames_and_drops_leaky_column():
    raw = pd.DataFrame({"emp.var.rate": [1.0], "duration": [100], "y": ["yes"]})
    out = clean(raw, ["duration"])
    assert "emp_var_rate" in out and "duration" not in out
    assert out[TARGET].tolist() == [1]


def test_validate_rejects_bad_data(synthetic):
    validate(synthetic)
    with pytest.raises(ValueError, match="missing"):
        validate(synthetic.drop(columns=["age"]))
    bad = synthetic.copy()
    bad.loc[0, "age"] = 300
    with pytest.raises(ValueError, match="age"):
        validate(bad)


@pytest.mark.parametrize(
    "name,hp",
    [("logreg", {"C": 1.0}), ("random_forest", {"n_estimators": 20, "min_samples_leaf": 5})],
)
def test_pipelines_fit_and_handle_unseen_category(synthetic, name, hp):
    model = build(name, hp, seed=0).fit(synthetic[FEATURES], synthetic[TARGET])
    new = synthetic[FEATURES].head(2).copy()
    new["job"] = "never-seen"
    proba = model.predict_proba(new)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_business_metrics():
    y = np.array([1, 0, 0, 1])
    p = np.array([0.9, 0.1, 0.6, 0.8])
    assert (
        expected_profit(y, p, 0.5, cost_per_call=5, revenue_per_subscription=50) == 2 * 50 - 3 * 5
    )
    thr, profit, _, _ = best_threshold(y, p, 5, 50)
    assert 0.6 < thr <= 0.8 and profit == 90
    assert lift_at(y, p, 0.5) == 2.0


def test_api_predict(synthetic, tmp_path, monkeypatch):
    model = build("logreg", {"C": 1.0}, seed=0).fit(synthetic[FEATURES], synthetic[TARGET])
    path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "threshold": 0.5, "features": FEATURES, "winner": "logreg"}, path)
    monkeypatch.setenv("MODEL_PATH", str(path))

    from dsp.bank import api

    api.bundle.cache_clear()
    client = TestClient(api.app)
    assert client.get("/health").json()["model"] == "logreg"

    row = synthetic[FEATURES].iloc[0].to_dict()
    row.update(
        default="no",
        housing="yes",
        loan="no",
        contact="cellular",
        poutcome="nonexistent",
        age=40,
        campaign=1,
        previous=0,
        pdays=999,
    )
    r = client.post("/predict", json=[row])
    assert r.status_code == 200
    assert 0 <= r.json()[0]["probability"] <= 1

    assert client.post("/predict", json=[{**row, "age": 5}]).status_code == 422
