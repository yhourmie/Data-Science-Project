# Data Science Portfolio: end-to-end ML projects

| # | Project | Type | Status |
|---|---------|------|--------|
| 1 | Bank marketing response prediction | Tabular ML (scikit-learn, XGBoost) | built |
| 2 | Washington EV type: plug-in hybrid or fully electric? | Tabular ML (gradient boosting, target encoding) | built |
| 3 | German electricity-load forecasting | Deep learning (PyTorch) | planned |
| 4 | Satellite land-use classification (EuroSAT) | Computer vision (PyTorch) | planned |

Shared tooling: DVC pipelines, MLflow experiment tracking and model registry, FastAPI serving,
Docker, pytest and GitHub Actions CI. Everything trains on a laptop CPU.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[bank,serve,dev]"
dvc repro                     # download data, train and evaluate every project
mlflow ui --backend-store-uri sqlite:///mlflow.db
uvicorn dsp.bank.api:app      # http://localhost:8000/docs (or dsp.ev.api:app)
```

## Docker

```bash
docker compose up -d mlflow                 # MLflow server: http://localhost:5000
docker compose run --rm bank-train          # download data + train in a container, logged to MLflow
docker compose up -d --build bank-api       # serve the new model: http://localhost:8001/docs
docker compose run --rm ev-train            # same for the EV project (about 20 min on a laptop CPU)
docker compose up -d --build ev-api         # http://localhost:8002/docs
```

One `docker/Dockerfile` serves every project, chosen with `--build-arg PROJECT=bank|ev`.
The training containers mount `data/`, `models/` and `reports/`, so results land in the repo just as
`dvc repro` would produce them. Models are also registered in MLflow as `bank-marketing` and `ev-phev`.
Experiments live in `mlflow-docker/`, separate from the local `mlflow.db`.

## EV project: where do drivers still choose plug-in hybrids?

Given only where an EV is registered (county, city, ZIP, legislative district, electric utility,
coordinates) and its model year, predict whether it is a plug-in hybrid (PHEV) or fully electric
(BEV). Make, model and electric range are excluded because they decide the answer outright.
PHEVs are 19% of Washington's roughly 300k registered EVs. Their share runs from about 16% in
King and Snohomish counties to over 40% in some rural counties, which points to places where
drivers still hedge against sparse charging.

High-cardinality ZIPs and cities use a cross-fitted `TargetEncoder`. Logistic regression,
HistGradientBoosting and XGBoost are compared with 5-fold CV, and the decision threshold is
tuned for F1 on out-of-fold predictions. Results (`reports/ev/metrics.json`) on the held-out
test set: ROC-AUC 0.665 and PR-AUC 0.356 against a 0.19 base rate. Location alone gives modest
per-vehicle signal, but predicted PHEV share per county tracks the observed share closely
(`reports/ev/county_share.png`).

## Data sources

- **Bank marketing:** [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
  (`bank-additional-full`), CC BY 4.0. Moro, Cortez & Rita (2014), *A data-driven approach to
  predict the success of bank telemarketing*, Decision Support Systems 62:22-31.
- **EV:** [Electric Vehicle Population Data](https://catalog.data.gov/dataset/electric-vehicle-population-data),
  Washington State Department of Licensing via data.wa.gov, Open Database License (ODbL).
  The file is refreshed regularly, so retraining takes a new snapshot.

Raw data is not committed; the pipelines download it.
