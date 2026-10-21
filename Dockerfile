# Search service. The corpus and models are mounted at run time, not baked in: page text is
# 190 MB and the embeddings 140 MB, and `make data` and `make embed` rebuild both from public
# sources.
#
#   docker build -t finrag .
#   docker run --rm -p 8000:8000 -v "$PWD/data:/data" -v "$PWD/models:/models:ro" finrag
#
# Without /models or /data/embeddings the service runs BM25 only and /healthz says so.
FROM python:3.12-slim@sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    FINRAG_DATA_DIR=/data FINRAG_ALIASES=/app/configs/aliases.yaml \
    FINRAG_MODEL_DIR=/models/all-MiniLM-L6-v2 FINRAG_EMB_DIR=/data/embeddings

WORKDIR /app
COPY requirements.lock .
RUN pip install --require-hashes -r requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .
COPY configs ./configs

RUN useradd --create-home --uid 1000 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
CMD ["uvicorn", "finrag.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
