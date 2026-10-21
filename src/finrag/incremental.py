"""Check that a BM25 index grown filing batch by filing batch matches one built in one pass.

The check holds out every tenth filing, builds an index over the rest and saves it, then
adds the held-out filings in batches, saving and reloading after each one the way
``load_or_build_bm25`` does when new filings arrive. It then scores every question against
both indexes, as asked and with the company's name dropped, over every page.

Both indexes run the same formula on the same statistics and add up each page's terms in
the same order, so the scores should be identical. The check allows a difference of
``TOLERANCE`` so that a harmless change in the order of floating-point operations doesn't
fail it, while a wrong count still does: one page missing from N moves every idf by about
2e-5 (1 / (N + 1)), four orders of magnitude above the tolerance.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from .bm25 import BM25, tokenize
from .data import Question, pages_digest
from .filters import MetadataFilter
from .search import Corpus, top_k

TOLERANCE = 1e-9
TOP = 100


def _tokens(corpus: Corpus, docs: list[str]) -> dict[str, list[list[str]]]:
    return {d: [tokenize(t) for t in corpus.pages(d)] for d in docs}


def check(corpus: Corpus, questions: list[Question], flt: MetadataFilter, holdout_every: int = 10,
          batch: int = 12) -> dict:
    hashes = {d: pages_digest(corpus.pages(d)) for d in corpus.docs}
    t0 = time.perf_counter()
    full = BM25()
    full.add(_tokens(corpus, corpus.docs), hashes)
    full_seconds = time.perf_counter() - t0
    full.assign_rows(corpus.offsets)

    held = corpus.docs[::holdout_every]
    base = [d for d in corpus.docs if d not in set(held)]
    add_seconds = []
    with tempfile.TemporaryDirectory() as tmp:
        grown = BM25()
        grown.add(_tokens(corpus, base), hashes)
        grown.save(Path(tmp))
        for i in range(0, len(held), batch):
            t0 = time.perf_counter()
            grown = BM25.load(Path(tmp))
            grown.add(_tokens(corpus, held[i : i + batch]), hashes)
            grown.save(Path(tmp))
            add_seconds.append(time.perf_counter() - t0)
        grown = BM25.load(Path(tmp))
    grown.assign_rows(corpus.offsets)

    queries = []
    for q in questions:
        tokens = tokenize(q.text)
        drop = flt.company_terms(flt.companies_in(q.text))
        queries += [tokens, [t for t in tokens if t not in drop]]
    max_diff = 0.0
    same_top = 0
    for tokens in queries:
        a, b = full.scores(tokens), grown.scores(tokens)
        max_diff = max(max_diff, float(np.abs(a - b).max()))
        same_top += bool(np.array_equal(top_k(a, TOP), top_k(b, TOP)))
    return {
        "filings": len(corpus.docs), "pages": len(corpus),
        "held_out_filings": len(held), "held_out_pages": sum(len(corpus.pages(d)) for d in held),
        "segments": len(grown.segments), "batch": batch,
        "queries": len(queries), "max_abs_score_diff": max_diff, "tolerance": TOLERANCE,
        "same_top100": same_top, "passed": max_diff <= TOLERANCE and same_top == len(queries),
        "full_build_seconds": round(full_seconds, 2),
        "add_batch_seconds": [round(s, 2) for s in add_seconds],
    }
