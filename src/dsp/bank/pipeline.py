"""Model pipelines and business-metric helpers for the bank project."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dsp.bank.data import CATEGORICAL, NUMERIC


def preprocessor(scale: bool) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", num, NUMERIC),
        ],
        verbose_feature_names_out=False,
    )


def build(name: str, hp: dict, seed: int, pos_weight: float = 1.0) -> Pipeline:
    if name == "logreg":
        clf = LogisticRegression(C=hp["C"], max_iter=2000, class_weight="balanced")
        return Pipeline([("prep", preprocessor(scale=True)), ("clf", clf)])
    if name == "random_forest":
        clf = RandomForestClassifier(
            **hp, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        )
        return Pipeline([("prep", preprocessor(scale=False)), ("clf", clf)])
    if name == "xgboost":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            **hp,
            scale_pos_weight=pos_weight,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
        return Pipeline([("prep", preprocessor(scale=False)), ("clf", clf)])
    raise ValueError(f"unknown model {name!r}")


def expected_profit(y_true, proba, threshold, cost_per_call, revenue_per_subscription) -> float:
    """Profit of calling every customer whose predicted probability >= threshold."""
    called = proba >= threshold
    return float(
        revenue_per_subscription * (called & (y_true == 1)).sum() - cost_per_call * called.sum()
    )


def best_threshold(y_true, proba, cost_per_call, revenue_per_subscription):
    grid = np.linspace(0.01, 0.99, 99)
    profits = [
        expected_profit(y_true, proba, t, cost_per_call, revenue_per_subscription) for t in grid
    ]
    i = int(np.argmax(profits))
    return float(grid[i]), float(profits[i]), grid, np.array(profits)


def lift_at(y_true, proba, frac: float = 0.1) -> float:
    n = max(1, int(len(proba) * frac))
    top = np.argsort(-proba)[:n]
    return float(np.asarray(y_true)[top].mean() / np.mean(y_true))
