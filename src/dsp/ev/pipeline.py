"""Model pipelines for the EV project.

ZIP codes and cities have hundreds of levels, so the tree models use a cross-fitted
TargetEncoder (no target leakage within a fit) instead of one-hot columns. The linear
baseline one-hot encodes, with rare levels pooled into an "infrequent" bucket.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from dsp.ev.data import CATEGORICAL, NUMERIC


def preprocessor(kind: str, seed: int, min_frequency: int = 20) -> ColumnTransformer:
    if kind == "onehot":
        cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency)
        num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    elif kind == "target":
        folds = StratifiedKFold(5, shuffle=True, random_state=seed)
        cat = TargetEncoder(target_type="binary", cv=folds)
        num = "passthrough"  # gradient-boosted trees handle missing coordinates natively
    else:
        raise ValueError(f"unknown preprocessor {kind!r}")
    return ColumnTransformer(
        [("cat", cat, CATEGORICAL), ("num", num, NUMERIC)], verbose_feature_names_out=False
    )


def build(name: str, hp: dict, seed: int, min_frequency: int = 20) -> Pipeline:
    """No class re-weighting: predicted probabilities stay calibrated, and the decision
    threshold is tuned separately on out-of-fold predictions."""
    if name == "logreg":
        clf = LogisticRegression(C=hp["C"], max_iter=2000)
        return Pipeline([("prep", preprocessor("onehot", seed, min_frequency)), ("clf", clf)])
    if name == "hist_gb":
        clf = HistGradientBoostingClassifier(**hp, random_state=seed)
        return Pipeline([("prep", preprocessor("target", seed)), ("clf", clf)])
    if name == "xgboost":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            **hp,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
        return Pipeline([("prep", preprocessor("target", seed)), ("clf", clf)])
    raise ValueError(f"unknown model {name!r}")


def best_f1_threshold(y_true, proba) -> tuple[float, float]:
    grid = np.linspace(0.01, 0.99, 99)
    scores = [f1_score(y_true, proba >= t, zero_division=0) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])
