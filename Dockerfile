# Search service. The corpus and model are mounted at runtime instead of baked in: page text
# is 190 MB, embeddings 140 MB, and both are rebuilt from public sources by `make data`.
#
#   docker build -t finrag .
#   docker run --rm -p 8000:8000 -v "$PWD/data:/data" -v "$PWD/models:/models:ro" finrag
#
# Without /models or /data/emb the service starts BM25-only and /healthz says so.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    FINRAG_DATA_DIR=/data FINRAG_ALIASES=/app/configs/aliases.yaml \
    FINRAG_MODEL_DIR=/models/all-MiniLM-L6-v2 FINRAG_EMB_DIR=/data/emb/minilm

WORKDIR /app
COPY requirements.lock .
RUN pip install -r requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .
COPY configs ./configs

RUN useradd --create-home --uid 1000 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
CMD ["uvicorn", "finrag.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
