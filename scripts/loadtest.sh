#!/usr/bin/env bash
# Load-test the search service with the server pinned to CPU 0 and the load generator to
# CPU 1, so the client doesn't steal the server's core. Needs Linux (taskset) and the full
# data: pages, embeddings, model.
set -euo pipefail
METHOD=${METHOD:-hybrid}
LEVELS=${LEVELS:-1,2,4,8,16,32}
SECONDS_PER_LEVEL=${SECONDS_PER_LEVEL:-30}
OUT=${OUT:-results/loadtest.json}

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export FINRAG_DATA_DIR=${FINRAG_DATA_DIR:-data} FINRAG_MODEL_DIR=${FINRAG_MODEL_DIR:-models/all-MiniLM-L6-v2}
export FINRAG_EMB_DIR=${FINRAG_EMB_DIR:-data/emb/minilm}
taskset -c 0 uvicorn finrag.service:app --host 127.0.0.1 --port 8000 --workers 1 --log-level warning &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do curl -fs 127.0.0.1:8000/healthz >/dev/null 2>&1 && break; sleep 1; done
curl -fs 127.0.0.1:8000/healthz
echo
taskset -c 1 finrag loadtest --url http://127.0.0.1:8000 --method "$METHOD" --levels "$LEVELS" \
  --seconds "$SECONDS_PER_LEVEL" --out "$OUT"
