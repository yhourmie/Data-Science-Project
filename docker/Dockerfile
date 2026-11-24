# One Dockerfile for every project; PROJECT picks which (bank, ev). Build from the repo root.
#   serve (default): docker build -f docker/Dockerfile --build-arg PROJECT=bank -t dsp-bank .
#                    (needs models/<PROJECT>, e.g. from `dvc repro <PROJECT>_train`)
#   train:           docker build -f docker/Dockerfile --build-arg PROJECT=bank --target train .
FROM python:3.12-slim AS base

ARG PROJECT=bank
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DSP_ROOT=/app \
    PROJECT=${PROJECT} GIT_PYTHON_REFRESH=quiet

COPY pyproject.toml README.md params.yaml ./
COPY src ./src


# Training: data/, models/ and reports/ are bind-mounted so outputs land in the repo.
FROM base AS train
RUN pip install ".[${PROJECT}]"
CMD python -m dsp.${PROJECT}.data && python -m dsp.${PROJECT}.train


# Serving (last stage, so it is the default target).
FROM base AS serve
ARG PROJECT
RUN pip install ".[${PROJECT},serve]"
COPY models/${PROJECT} ./models/${PROJECT}

RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD exec uvicorn dsp.${PROJECT}.api:app --host 0.0.0.0 --port 8000
