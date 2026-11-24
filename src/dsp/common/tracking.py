"""Shared helpers: paths, params, seeding and MLflow setup."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(os.environ.get("DSP_ROOT", Path(__file__).resolve().parents[3]))
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def load_params(section: str | None = None) -> dict[str, Any]:
    with open(ROOT / "params.yaml", encoding="utf-8") as fh:
        params = yaml.safe_load(fh)
    if section is None:
        return params
    return {"seed": params["seed"], **params[section]}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def setup_mlflow(experiment: str):
    """Point MLflow at $MLFLOW_TRACKING_URI, defaulting to a local SQLite store."""
    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return mlflow


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
