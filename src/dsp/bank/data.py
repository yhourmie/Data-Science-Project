"""Download, validate and split the UCI Bank Marketing dataset (bank-additional-full)."""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from dsp.common.tracking import DATA_DIR, load_params

RAW = DATA_DIR / "raw" / "bank"
PROCESSED = DATA_DIR / "processed" / "bank"
TARGET = "y"

CATEGORICAL = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]
NUMERIC = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor3m",
    "nr_employed",
]
FEATURES = CATEGORICAL + NUMERIC


def download(url: str) -> pd.DataFrame:
    """The UCI archive is a zip that contains another zip (bank-additional.zip)."""
    outer = zipfile.ZipFile(io.BytesIO(requests.get(url, timeout=120).content))
    inner = zipfile.ZipFile(io.BytesIO(outer.read("bank-additional.zip")))
    with inner.open("bank-additional/bank-additional-full.csv") as fh:
        return pd.read_csv(fh, sep=";")


def clean(df: pd.DataFrame, drop_columns: list[str]) -> pd.DataFrame:
    df = df.rename(columns=lambda c: c.replace(".", "_"))
    df = df.drop(columns=[c for c in drop_columns if c in df.columns])
    df[TARGET] = (df[TARGET] == "yes").astype(int)
    return df.drop_duplicates().reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    """Fail fast on schema drift instead of training on broken data."""
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if df[FEATURES].isna().any().any():
        raise ValueError("unexpected NaNs in features")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("target must be binary")
    if not df["age"].between(15, 110).all():
        raise ValueError("age out of plausible range")


def main() -> None:
    p = load_params("bank")
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    raw = download(p["data_url"])
    raw.to_csv(RAW / "bank-additional-full.csv", index=False)

    df = clean(raw, p["drop_columns"])
    validate(df)
    train, test = train_test_split(
        df, test_size=p["test_size"], stratify=df[TARGET], random_state=p["seed"]
    )
    train.to_csv(PROCESSED / "train.csv", index=False)
    test.to_csv(PROCESSED / "test.csv", index=False)
    print(
        f"rows={len(df)} train={len(train)} test={len(test)} positive_rate={df[TARGET].mean():.3f}"
    )


if __name__ == "__main__":
    main()
