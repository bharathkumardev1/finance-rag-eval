"""Page-level retrieval metrics against FinanceBench evidence.

A retrieved chunk is relevant when it comes from an evidence document and spans an evidence
page. For each question:

* ``hit@k``: at least one relevant chunk in the top k.
* ``recall@k``: share of the question's evidence pages covered by the top k.
* ``mrr@k``: reciprocal rank of the first relevant chunk, 0 if none in the top k.
* ``doc_hit@k``: at least one top-k chunk from an evidence document (right filing, any page).
"""

from __future__ import annotations

import numpy as np

from .chunking import Chunk
from .data import Question


def _evidence(q: Question) -> set[tuple[str, int]]:
    return {(e.doc_name, e.page) for e in q.evidence}


def score_question(ranking: list[int], chunks: list[Chunk], q: Question, ks=(1, 5, 10)) -> dict:
    ev = _evidence(q)
    ev_docs = {d for d, _ in ev}
    out: dict[str, float] = {}
    rel = []
    covered_at: list[set[tuple[str, int]]] = []
    seen: set[tuple[str, int]] = set()
    for idx in ranking:
        c = chunks[idx]
        hits = {(d, p) for d, p in ev if d == c.doc_name and p in c.pages}
        rel.append(bool(hits))
        seen |= hits
        covered_at.append(set(seen))
    first = next((i for i, r in enumerate(rel) if r), None)
    for k in ks:
        out[f"hit@{k}"] = float(any(rel[:k]))
        out[f"recall@{k}"] = len(covered_at[min(k, len(rel)) - 1]) / len(ev) if rel else 0.0
        out[f"doc_hit@{k}"] = float(any(chunks[i].doc_name in ev_docs for i in ranking[:k]))
    kmax = max(ks)
    out[f"mrr@{kmax}"] = 1.0 / (first + 1) if first is not None and first < kmax else 0.0
    return out


def aggregate(per_question: list[dict]) -> dict[str, float]:
    if not per_question:
        return {}
    keys = per_question[0].keys()
    return {k: float(np.mean([r[k] for r in per_question])) for k in keys}


def latency_summary(seconds: list[float]) -> dict[str, float]:
    if not seconds:
        return {"p50_ms": 0.0, "p95_ms": 0.0}
    arr = np.array(seconds) * 1000.0
    return {"p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95))}
