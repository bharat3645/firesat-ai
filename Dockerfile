# FireSat-AI API + dashboard container.
# Build:  docker build -t firesat-ai .
# Run:    docker run -p 8000:8000 firesat-ai
# (ships with the synthetic demo dataset + a trained demo checkpoint baked
#  in via data/processed/ and models/checkpoints/, so it works immediately;
#  mount your own volumes over those paths to serve real acquired data.)

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
COPY dashboard ./dashboard
COPY scripts ./scripts
COPY data/processed ./data/processed
COPY models/checkpoints ./models/checkpoints

RUN pip install --no-cache-dir -e .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/health').raise_for_status()" || exit 1

CMD ["uvicorn", "firesat.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
