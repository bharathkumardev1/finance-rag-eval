"""Chunk-size study: BM25 over fixed-size chunks instead of whole pages.

Each page's tokens are cut into chunks of ``size`` tokens that overlap by ``overlap``, never
crossing a page boundary, and BM25 indexes the chunks as its documents (so N, df and the
average length are chunk statistics). A page then scores as its best chunk, which keeps the
page-level metrics, the filter and the statement routing exactly as they are for pages.

Sizes are in BM25 tokens (stopwords removed). The median page has 352 of them and 97% of
pages have 650 or fewer, so 650-token chunks are mostly whole pages.
"""

from __future__ import annotations

import numpy as np

from .bm25 import BM25, tokenize
from .search import Corpus

SIZES = ((128, 32), (256, 64), (650, 100))


def chunk(tokens: list[str], size: int, overlap: int) -> list[list[str]]:
    if not 0 <= overlap < size:
        raise ValueError(f"need 0 <= overlap < size, got size={size}, overlap={overlap}")
    if len(tokens) <= size:
        return [tokens]
    step = size - overlap
    out = [tokens[i : i + size] for i in range(0, len(tokens) - overlap, step)]
    return [c for c in out if c]


class ChunkedBM25:
    """Stands in for ``BM25`` in a Searcher: same ``scores`` call, page scores from chunks."""

    def __init__(self, corpus: Corpus, size: int, overlap: int):
        chunks: list[list[str]] = []
        counts = np.zeros(len(corpus), dtype=np.int64)
        for row, text in enumerate(corpus.texts):
            pieces = chunk(tokenize(text), size, overlap)
            chunks.extend(pieces)
            counts[row] = len(pieces)
        self.index = BM25.from_pages(chunks)
        self.page_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self.n_chunks = len(chunks)
        self.size, self.overlap = size, overlap

    def scores(self, tokens: list[str], rows: np.ndarray | None = None) -> np.ndarray:
        pages = np.maximum.reduceat(self.index.scores(tokens), self.page_start)
        return pages if rows is None else pages[rows]


def study(searcher, questions, split, sizes=SIZES) -> dict:
    """Page retrieval against each chunk size, with and without the full pipeline around BM25."""
    import copy
    import time

    from .evaluate import compare, run
    from .search import ALL_PAGES, FULL

    units = {"pages": searcher.bm25}
    built = {"pages": {"documents": len(searcher.corpus)}}
    for size, overlap in sizes:
        t0 = time.perf_counter()
        index = ChunkedBM25(searcher.corpus, size, overlap)
        name = f"chunks-{size}-{overlap}"
        units[name] = index
        built[name] = {"documents": index.n_chunks, "build_seconds": round(time.perf_counter() - t0, 1)}

    def system(index, opts):
        s = copy.copy(searcher)
        s.bm25 = index
        return lambda text: s.ranked_pages(text, 100, "bm25", opts)

    systems = {f"bm25-{name}/{opts.key()}": system(index, opts)
               for name, index in units.items() for opts in (ALL_PAGES, FULL)}
    results = run(systems, questions, split)
    results["units"] = built
    results["comparisons"] = [
        compare(results, f"bm25-{name}/{opts.key()}", f"bm25-pages/{opts.key()}", "test", metric)
        for name in units if name != "pages" for opts in (ALL_PAGES, FULL)
        for metric in ("hit@10", "mrr@10")]
    return results
