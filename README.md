# Data Science Portfolio: three end-to-end ML projects

| # | Project | Type | Status |
|---|---------|------|--------|
| 1 | Bank marketing response prediction | Tabular ML (scikit-learn, XGBoost) | built |
| 2 | German electricity-load forecasting | Deep learning (PyTorch) | planned |
| 3 | Satellite land-use classification (EuroSAT) | Computer vision (PyTorch) | planned |

Shared tooling: DVC pipelines, MLflow experiment tracking and model registry, FastAPI serving,
Docker, pytest and GitHub Actions CI. Everything trains on a laptop CPU.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[bank,serve,dev]"
dvc repro bank_train          # download data, train, evaluate
mlflow ui --backend-store-uri sqlite:///mlflow.db
uvicorn dsp.bank.api:app      # http://localhost:8000/docs
```

## Docker

```bash
docker compose up -d mlflow                 # MLflow server: http://localhost:5000
docker compose run --rm bank-train          # download data + train in a container, logged to MLflow
docker compose up -d --build bank-api       # serve the new model: http://localhost:8001/docs
```

`bank-train` mounts `data/`, `models/` and `reports/`, so results land in the repo just as
`dvc repro` would produce them. The model also appears in the MLflow registry as `bank-marketing`.
Experiments live in `mlflow-docker/`, separate from the local `mlflow.db`.
