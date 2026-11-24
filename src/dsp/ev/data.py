"""Download, clean and split the Washington State Electric Vehicle Population Data.

Source: Washington State Department of Licensing via data.wa.gov (dataset f6w7-q2d2), listed
on catalog.data.gov. Licence: Open Data Commons Open Database License (ODbL). The file is
refreshed regularly, so each `dvc repro ev_data` takes a new snapshot.

Task: is a registered EV a plug-in hybrid (1) or fully electric (0), given only where it is
registered and its model year? Make, model and electric range are dropped because they decide
the answer outright.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from dsp.common.tracking import DATA_DIR, load_params

RAW = DATA_DIR / "raw" / "ev"
PROCESSED = DATA_DIR / "processed" / "ev"
TARGET = "phev"

CATEGORICAL = ["county", "city", "postal_code", "legislative_district", "electric_utility"]
NUMERIC = ["model_year", "longitude", "latitude"]
FEATURES = CATEGORICAL + NUMERIC

RENAME = {
    "County": "county",
    "City": "city",
    "State": "state",
    "Postal Code": "postal_code",
    "Model Year": "model_year",
    "Electric Vehicle Type": "ev_type",
    "Legislative District": "legislative_district",
    "Vehicle Location": "vehicle_location",
    "Electric Utility": "electric_utility",
    "DOL Vehicle ID": "dol_vehicle_id",
}
PHEV = "Plug-in Hybrid Electric Vehicle (PHEV)"
BEV = "Battery Electric Vehicle (BEV)"


def download(url: str, dest) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)


def _code(s: pd.Series) -> pd.Series:
    """ZIP and district codes arrive as floats when the column has gaps: 98052.0 -> '98052'."""

    def fmt(v) -> str:
        if pd.isna(v):
            return "unknown"
        return str(int(v)) if isinstance(v, int | float | np.number) else str(v)

    return s.map(fmt)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=RENAME)
    df = df[(df["state"] == "WA") & df["ev_type"].isin([PHEV, BEV])]
    df = df.drop_duplicates("dol_vehicle_id")

    lonlat = df["vehicle_location"].str.extract(r"POINT \((?P<lon>[-\d.]+) (?P<lat>[-\d.]+)\)")
    out = pd.DataFrame(
        {
            "county": df["county"].fillna("unknown"),
            "city": df["city"].fillna("unknown").str.upper(),
            "postal_code": _code(df["postal_code"]),
            "legislative_district": _code(df["legislative_district"]),
            "electric_utility": df["electric_utility"].fillna("unknown"),
            "model_year": df["model_year"].astype(int),
            "longitude": pd.to_numeric(lonlat["lon"]),
            "latitude": pd.to_numeric(lonlat["lat"]),
            TARGET: (df["ev_type"] == PHEV).astype(int),
        }
    )
    return out.reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    """Fail fast on schema drift instead of training on broken data."""
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if df[CATEGORICAL].isna().any().any():
        raise ValueError("unexpected NaNs in categorical features")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("target must be binary")
    if not df["model_year"].between(1990, 2035).all():
        raise ValueError("model_year out of plausible range")
    lon, lat = df["longitude"].dropna(), df["latitude"].dropna()
    # Generous box around Washington State; a few registrations sit just over the border.
    if not (lon.between(-126, -115).all() and lat.between(44, 50).all()):
        raise ValueError("coordinates outside the Washington area")
    if not 0.01 < df[TARGET].mean() < 0.99:
        raise ValueError("target has (almost) a single class")


def main() -> None:
    p = load_params("ev")
    raw_path = RAW / "ev_population.csv"
    download(p["data_url"], raw_path)

    df = clean(pd.read_csv(raw_path, low_memory=False))
    validate(df)
    train, test = train_test_split(
        df, test_size=p["test_size"], stratify=df[TARGET], random_state=p["seed"]
    )
    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED / "train.csv", index=False)
    test.to_csv(PROCESSED / "test.csv", index=False)
    print(f"rows={len(df)} train={len(train)} test={len(test)} phev_rate={df[TARGET].mean():.3f}")


if __name__ == "__main__":
    main()
