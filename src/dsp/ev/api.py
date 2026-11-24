"""FastAPI service: POST where an EV is registered and its model year, get P(plug-in hybrid)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from dsp.common.tracking import MODELS_DIR

app = FastAPI(title="Washington EV type model", version="1.0")


class Vehicle(BaseModel):
    county: str = Field(examples=["King"])
    city: str = Field(examples=["SEATTLE"])
    postal_code: str = Field(examples=["98101"])
    legislative_district: str = Field(examples=["43"])
    electric_utility: str = Field(examples=["CITY OF SEATTLE - (WA)|CITY OF TACOMA - (WA)"])
    model_year: int = Field(ge=1990, le=2035, examples=[2023])
    longitude: float | None = Field(default=None, ge=-126, le=-115, examples=[-122.33])
    latitude: float | None = Field(default=None, ge=44, le=50, examples=[47.61])


class Prediction(BaseModel):
    phev_probability: float
    predicted_type: Literal["PHEV", "BEV"]
    threshold: float


@lru_cache(maxsize=1)
def bundle() -> dict:
    return joblib.load(os.environ.get("MODEL_PATH", MODELS_DIR / "ev" / "model.joblib"))


@app.get("/health")
def health() -> dict:
    b = bundle()
    return {"status": "ok", "model": b["winner"], "threshold": b["threshold"]}


@app.post("/predict", response_model=list[Prediction])
def predict(vehicles: list[Vehicle]) -> list[Prediction]:
    b = bundle()
    X = pd.DataFrame([v.model_dump() for v in vehicles])[b["features"]]
    X["city"] = X["city"].str.upper()
    X[["longitude", "latitude"]] = X[["longitude", "latitude"]].astype(float)
    proba = b["model"].predict_proba(X)[:, 1]
    return [
        Prediction(
            phev_probability=float(pr),
            predicted_type="PHEV" if pr >= b["threshold"] else "BEV",
            threshold=b["threshold"],
        )
        for pr in proba
    ]
