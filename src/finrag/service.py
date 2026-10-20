"""HTTP search service: ``GET /search``, ``GET /healthz``, ``GET /metrics``.

Configuration comes from the environment so the same image runs anywhere:

* ``FINRAG_DATA_DIR`` (default ``data``): page cache, filing metadata, BM25 index cache
* ``FINRAG_ALIASES`` (default ``configs/aliases.yaml``)
* ``FINRAG_MODEL_DIR`` and ``FINRAG_EMB_DIR``: MiniLM model and page embeddings. If either
  is missing the service starts with BM25 only and says so in ``/healthz``.

The handlers are plain ``def`` functions, so FastAPI runs them on its thread pool. NumPy and
ONNX Runtime release the GIL for the heavy parts; run one worker per core for throughput.
"""

from __future__ import annotations

import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .bm25 import tokenize
from .search import METHODS, Options, Searcher, load_aliases

log = logging.getLogger("finrag.service")

REQUESTS = Counter("finrag_requests_total", "Search requests", ["method", "status"])
LATENCY = Histogram(
    "finrag_search_seconds", "Search latency inside the handler", ["method"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
NO_COMPANY = Counter("finrag_filter_no_company_total",
                     "Queries where the filter found no company and search ran unfiltered")
ROUTED = Counter("finrag_statement_routed_total", "Queries routed to statement pages")

SNIPPET_CHARS = 280


_WORD = re.compile(r"(?u)\b\w\w+\b")


def snippet(text: str, question: str, width: int = SNIPPET_CHARS) -> str:
    """The ``width``-character window of the page covering the most distinct query words."""
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    words = set(tokenize(question))
    hits = [(m.start(), m.group()) for m in _WORD.finditer(flat.lower()) if m.group() in words]
    best_start, best_count = 0, 0
    counts: dict[str, int] = {}
    j = 0
    for pos, word in hits:  # two pointers over hit positions
        while j < len(hits) and hits[j][0] < pos + width - 40:
            counts[hits[j][1]] = counts.get(hits[j][1], 0) + 1
            j += 1
        if len(counts) > best_count:
            best_start, best_count = pos, len(counts)
        counts[word] -= 1
        if not counts[word]:
            del counts[word]
    start = max(0, best_start - 40)
    return ("…" if start else "") + flat[start : start + width] + "…"


def load_searcher() -> Searcher:
    data_dir = Path(os.environ.get("FINRAG_DATA_DIR", "data"))
    aliases_path = Path(os.environ.get("FINRAG_ALIASES", "configs/aliases.yaml"))
    model_dir = os.environ.get("FINRAG_MODEL_DIR")
    emb_dir = os.environ.get("FINRAG_EMB_DIR")
    dense_ok = bool(model_dir and emb_dir and Path(model_dir).exists() and Path(emb_dir).exists())
    t0 = time.perf_counter()
    searcher = Searcher.build(
        data_dir,
        load_aliases(aliases_path) if aliases_path.exists() else {},
        Path(model_dir) if dense_ok else None,
        Path(emb_dir) if dense_ok else None,
        threads=int(os.environ.get("FINRAG_THREADS", "1")),
    )
    log.info("loaded %d pages, dense=%s in %.1fs", len(searcher.corpus), dense_ok,
             time.perf_counter() - t0)
    return searcher


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "searcher"):
        app.state.searcher = load_searcher()
    yield


app = FastAPI(title="finrag", version="2.0", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict:
    s: Searcher = app.state.searcher
    return {"status": "ok", "filings": len(s.corpus.docs), "pages": len(s.corpus),
            "methods": list(s.methods)}


@app.get("/search")
def search(
    q: str = Query(..., min_length=3, max_length=2000, description="the question"),
    method: str = Query("hybrid", description="bm25 | dense | hybrid"),
    k: int = Query(10, ge=1, le=50),
    filter: str = Query("company_year", description="none | company | company_year"),
    route: bool = Query(True, description="send statement questions to statement pages"),
) -> dict:
    s: Searcher = app.state.searcher
    if method not in METHODS:
        REQUESTS.labels(method="invalid", status="400").inc()
        raise HTTPException(400, f"method must be one of {list(METHODS)}")
    if method not in s.methods:
        REQUESTS.labels(method=method, status="503").inc()
        raise HTTPException(503, f"{method} search is not loaded on this instance")
    if filter not in ("none", "company", "company_year"):
        REQUESTS.labels(method=method, status="400").inc()
        raise HTTPException(400, "filter must be none, company or company_year")
    t0 = time.perf_counter()
    try:
        hits, info = s.search(q, method, k, Options(filter=filter, route_statements=route))
    except Exception:
        REQUESTS.labels(method=method, status="500").inc()
        log.exception("search failed for %r", q)
        raise
    LATENCY.labels(method=method).observe(time.perf_counter() - t0)
    REQUESTS.labels(method=method, status="200").inc()
    if filter != "none" and info["filings"] is None:
        NO_COMPANY.inc()
    if info["statements"]:
        ROUTED.inc()
    c = s.corpus
    return {
        **info,
        "hits": [{"doc": h.doc, "page": h.page, "score": round(h.score, 5),
                  "statement": sorted(s.kinds[h.row]), "snippet": snippet(c.texts[h.row], q)}
                 for h in hits],
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
