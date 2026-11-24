# Bank-marketing images, built from the repo root.
#   serve (default): docker build -f docker/bank.Dockerfile -t dsp-bank .   (needs models/bank)
#   train:           docker build -f docker/bank.Dockerfile --target train -t dsp-bank-train .
FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DSP_ROOT=/app

COPY pyproject.toml README.md params.yaml ./
COPY src ./src


# Training: data/, models/ and reports/ are bind-mounted so outputs land in the repo.
FROM base AS train
RUN pip install ".[bank]"
CMD ["sh", "-c", "python -m dsp.bank.data && python -m dsp.bank.train"]


# Serving (last stage, so it is the default target).
FROM base AS serve
RUN pip install ".[bank,serve]"
COPY models/bank ./models/bank

RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "dsp.bank.api:app", "--host", "0.0.0.0", "--port", "8000"]
