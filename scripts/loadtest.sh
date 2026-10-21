#!/usr/bin/env bash
# Load test on a 2-CPU machine: the service pinned to CPU 0, the load generator to CPU 1, so
# the two don't compete. Needs `make data` and `make embed` first (hybrid needs embeddings).
set -euo pipefail

export FINRAG_DATA_DIR=${FINRAG_DATA_DIR:-data}
export FINRAG_MODEL_DIR=${FINRAG_MODEL_DIR:-models/all-MiniLM-L6-v2}
export FINRAG_EMB_DIR=${FINRAG_EMB_DIR:-data/embeddings}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

taskset -c 0 uvicorn finrag.service:app --host 127.0.0.1 --port 8000 --workers 1 --log-level warning &
server=$!
trap 'kill $server' EXIT
until curl -sf http://127.0.0.1:8000/healthz >/dev/null; do sleep 1; done

taskset -c 1 finrag loadtest --methods bm25,hybrid --levels 1,2,4,8,16 --seconds 30 \
  --server-note "1 uvicorn worker pinned to CPU 0, load generator on CPU 1"
