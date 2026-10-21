"""Page search over the corpus with BM25, MiniLM, or both fused with reciprocal rank fusion.

Pages are numbered in one global order (filings sorted by name, pages in filing order), so
any set of filings is a set of contiguous row ranges and a retriever can score just those
rows.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25, tokenize
from .data import DocMeta, load_doc_meta, load_pages

METHODS = ("bm25", "dense", "hybrid")
RRF_K = 60
RRF_DEPTH = 100


@dataclass(frozen=True)
class Hit:
    row: int
    doc: str
    page: int
    score: float


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Positions of the k highest scores, best first; ties go to the lower position."""
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.lexsort((part, -scores[part]))]


def rrf(rankings: list[np.ndarray], k: int = RRF_K) -> tuple[np.ndarray, np.ndarray]:
    """Reciprocal rank fusion: a row scores sum(1 / (k + rank)) over the lists it appears in.

    Returns rows and fused scores, best first; ties go to the lower row.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking.tolist(), start=1):
            fused[row] = fused.get(row, 0.0) + 1.0 / (k + rank)
    rows = np.fromiter(fused, dtype=np.int64, count=len(fused))
    scores = np.fromiter(fused.values(), dtype=np.float64, count=len(fused))
    order = np.lexsort((rows, -scores))
    return rows[order], scores[order]


class Corpus:
    """Page text and filing metadata in global row order."""

    def __init__(self, pages: dict[str, list[str]], metas: dict[str, DocMeta]):
        self.docs = sorted(pages)
        self.metas = metas
        self.offsets: dict[str, tuple[int, int]] = {}
        self.texts: list[str] = []
        self.row_doc: list[str] = []
        self.row_page: list[int] = []
        for doc in self.docs:
            start = len(self.texts)
            self.texts.extend(pages[doc])
            self.row_doc.extend([doc] * len(pages[doc]))
            self.row_page.extend(range(len(pages[doc])))
            self.offsets[doc] = (start, len(self.texts))

    @classmethod
    def load(cls, data_dir: Path) -> Corpus:
        return cls(load_pages(data_dir), load_doc_meta(data_dir))

    def key(self, row: int) -> tuple[str, int]:
        return self.row_doc[row], self.row_page[row]

    def digest(self) -> str:
        """Hash of the page text, so cached indexes are rebuilt when the text changes."""
        h = hashlib.sha256()
        for doc in self.docs:
            h.update(doc.encode())
            start, end = self.offsets[doc]
            for t in self.texts[start:end]:
                h.update(b"\f" + t.encode("utf8"))
        return h.hexdigest()[:16]

    def __len__(self) -> int:
        return len(self.texts)


def load_or_build_bm25(corpus: Corpus, cache_dir: Path) -> BM25:
    prefix = Path(cache_dir) / f"bm25-{corpus.digest()}"
    if Path(f"{prefix}.npz").exists() and Path(f"{prefix}.json").exists():
        return BM25.load(prefix)
    index = BM25(tokenize(t) for t in corpus.texts)
    try:
        prefix.parent.mkdir(parents=True, exist_ok=True)
        index.save(prefix)
    except OSError:  # a read-only data volume keeps the index in memory only
        pass
    return index


class Searcher:
    def __init__(self, corpus: Corpus, bm25: BM25, dense=None, encoder=None):
        self.corpus = corpus
        self.bm25 = bm25
        self.dense = dense
        self.encoder = encoder

    @classmethod
    def build(cls, data_dir: Path, model_dir: Path | None = None, emb_dir: Path | None = None,
              threads: int = 1) -> Searcher:
        corpus = Corpus.load(data_dir)
        bm25 = load_or_build_bm25(corpus, Path(data_dir) / "index")
        dense = encoder = None
        if model_dir is not None and emb_dir is not None:
            from .dense import DenseIndex, MiniLM

            dense = DenseIndex.load(emb_dir, corpus.offsets)
            encoder = MiniLM(model_dir, threads=threads)
        return cls(corpus, bm25, dense, encoder)

    @property
    def methods(self) -> tuple[str, ...]:
        return METHODS if self.dense is not None else ("bm25",)

    def _scores(self, question: str, method: str, qvec: np.ndarray | None) -> np.ndarray:
        if method == "bm25":
            return self.bm25.scores(tokenize(question))
        if self.dense is None:
            raise RuntimeError("no dense index loaded; run `finrag embed` and pass --model-dir")
        if qvec is None:
            qvec = self.encoder.encode([question])[0]
        return self.dense.scores(qvec)

    def rank(self, question: str, k: int, method: str = "bm25") -> tuple[np.ndarray, np.ndarray]:
        """Top-k page rows over the whole corpus, best first, with their scores."""
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
        if method != "hybrid":
            scores = self._scores(question, method, None)
            order = top_k(scores, k)
            return order, scores[order].astype(np.float64)
        qvec = self.encoder.encode([question])[0] if self.encoder is not None else None
        lists = [top_k(self._scores(question, m, qvec), RRF_DEPTH) for m in ("bm25", "dense")]
        rows, scores = rrf(lists)
        return rows[:k], scores[:k]

    def ranked_pages(self, question: str, k: int, method: str = "bm25") -> list[tuple[str, int]]:
        rows, _ = self.rank(question, k, method)
        return [self.corpus.key(int(r)) for r in rows]

    def search(self, question: str, k: int = 10, method: str = "bm25") -> tuple[list[Hit], dict]:
        t0 = time.perf_counter()
        rows, scores = self.rank(question, k, method)
        hits = [Hit(int(r), *self.corpus.key(int(r)), float(s)) for r, s in zip(rows, scores, strict=True)]
        return hits, {"method": method, "pages_searched": len(self.corpus),
                      "ms": round((time.perf_counter() - t0) * 1e3, 3)}
