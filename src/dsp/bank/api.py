"""FastAPI service: POST customer features, get subscription probability and a call decision."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from dsp.common.tracking import MODELS_DIR

app = FastAPI(title="Bank marketing response model", version="1.0")


class Customer(BaseModel):
    age: int = Field(ge=15, le=110, examples=[41])
    job: str = Field(examples=["admin."])
    marital: str = Field(examples=["married"])
    education: str = Field(examples=["university.degree"])
    default: Literal["no", "yes", "unknown"] = "no"
    housing: Literal["no", "yes", "unknown"] = "yes"
    loan: Literal["no", "yes", "unknown"] = "no"
    contact: Literal["cellular", "telephone"] = "cellular"
    month: str = Field(examples=["may"])
    day_of_week: str = Field(examples=["mon"])
    campaign: int = Field(ge=1, examples=[2])
    pdays: int = Field(examples=[999], description="999 = not previously contacted")
    previous: int = Field(ge=0, examples=[0])
    poutcome: Literal["failure", "nonexistent", "success"] = "nonexistent"
    emp_var_rate: float = Field(examples=[1.1])
    cons_price_idx: float = Field(examples=[93.994])
    cons_conf_idx: float = Field(examples=[-36.4])
    euribor3m: float = Field(examples=[4.857])
    nr_employed: float = Field(examples=[5191.0])


class Prediction(BaseModel):
    probability: float
    call: bool
    threshold: float


@lru_cache(maxsize=1)
def bundle() -> dict:
    return joblib.load(os.environ.get("MODEL_PATH", MODELS_DIR / "bank" / "model.joblib"))


@app.get("/health")
def health() -> dict:
    b = bundle()
    return {"status": "ok", "model": b["winner"], "threshold": b["threshold"]}


@app.post("/predict", response_model=list[Prediction])
def predict(customers: list[Customer]) -> list[Prediction]:
    b = bundle()
    X = pd.DataFrame([c.model_dump() for c in customers])[b["features"]]
    proba = b["model"].predict_proba(X)[:, 1]
    return [
        Prediction(probability=float(pr), call=bool(pr >= b["threshold"]), threshold=b["threshold"])
        for pr in proba
    ]
