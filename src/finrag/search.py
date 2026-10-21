"""Page search: a metadata filter, then BM25, MiniLM, or both fused with reciprocal rank fusion.

Pages are numbered in one global order (filings sorted by name, pages in filing order), so a
filter result is a set of contiguous row ranges and both retrievers score just those rows.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25, tokenize
from .data import DocMeta, load_doc_meta, load_pages
from .filters import MetadataFilter
from .statements import page_kinds, question_kinds, route

METHODS = ("bm25", "dense", "hybrid")
RRF_K = 60
RRF_DEPTH = 100


@dataclass(frozen=True)
class Options:
    """What runs around the retriever. The defaults search every page with nothing around it."""

    filter: str = "none"               # none | company | company_year
    quarters: bool = False             # company_year also narrows to a quarter's filing
    drop_company_terms: bool = False   # BM25 query without the company's name once filtered
    route_statements: bool = False     # statement questions go to statement pages first

    def key(self) -> str:
        if self.filter == "none":
            return "all-pages"
        parts = [self.filter]
        if self.filter == "company_year" and self.quarters:
            parts.append("quarter")
        if self.drop_company_terms:
            parts.append("drop")
        if self.route_statements:
            parts.append("route")
        return "+".join(parts)


ALL_PAGES = Options()
FULL = Options("company_year", quarters=True, drop_company_terms=True, route_statements=True)


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

    def rows_for(self, docs: set[str]) -> np.ndarray:
        ranges = [np.arange(*self.offsets[d]) for d in sorted(docs) if d in self.offsets]
        return np.concatenate(ranges) if ranges else np.empty(0, dtype=np.int64)

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
    def __init__(self, corpus: Corpus, bm25: BM25, flt: MetadataFilter | None = None,
                 dense=None, encoder=None):
        self.corpus = corpus
        self.bm25 = bm25
        self.filter = flt or MetadataFilter(corpus.metas)
        self.dense = dense
        self.encoder = encoder
        self.kinds = [page_kinds(t) for t in corpus.texts]

    @classmethod
    def build(cls, data_dir: Path, aliases: dict[str, list[str]] | None = None,
              model_dir: Path | None = None, emb_dir: Path | None = None,
              threads: int = 1) -> Searcher:
        corpus = Corpus.load(data_dir)
        bm25 = load_or_build_bm25(corpus, Path(data_dir) / "index")
        flt = MetadataFilter(corpus.metas, aliases or {})
        dense = encoder = None
        if model_dir is not None and emb_dir is not None:
            from .dense import DenseIndex, MiniLM

            dense = DenseIndex.load(emb_dir, corpus.offsets)
            encoder = MiniLM(model_dir, threads=threads)
        return cls(corpus, bm25, flt, dense, encoder)

    @property
    def methods(self) -> tuple[str, ...]:
        return METHODS if self.dense is not None else ("bm25",)

    def filtered(self, question: str, opts: Options) -> tuple[set[str] | None, np.ndarray | None]:
        """Filings the filter keeps and their page rows; (None, None) means every page."""
        docs = self.filter.allowed_docs(question, opts.filter, opts.quarters)
        return (None, None) if docs is None else (docs, self.corpus.rows_for(docs))

    def query_tokens(self, question: str, filtered: bool, opts: Options) -> list[str]:
        tokens = tokenize(question)
        if filtered and opts.drop_company_terms:
            drop = self.filter.company_terms(self.filter.companies_in(question))
            tokens = [t for t in tokens if t not in drop]
        return tokens

    def _scores(self, question: str, method: str, rows: np.ndarray | None, opts: Options,
                qvec: np.ndarray | None) -> np.ndarray:
        if method == "bm25":
            return self.bm25.scores(self.query_tokens(question, rows is not None, opts), rows)
        if self.dense is None:
            raise RuntimeError("no dense index loaded; run `finrag embed` and pass --model-dir")
        if qvec is None:
            qvec = self.encoder.encode([question])[0]
        return self.dense.scores(qvec, rows)

    def rank(self, question: str, k: int, method: str = "bm25", opts: Options = ALL_PAGES
             ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k page rows and their scores, best first."""
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
        _, rows = self.filtered(question, opts)
        # With a filter the candidates are a few hundred pages, so rank all of them; that way
        # statement routing can promote a page however low the retriever put it.
        depth = k if rows is None else len(rows)
        if method == "hybrid":
            qvec = self.encoder.encode([question])[0] if self.encoder is not None else None
            lists = []
            for m in ("bm25", "dense"):
                order = top_k(self._scores(question, m, rows, opts, qvec), RRF_DEPTH)
                lists.append(order if rows is None else rows[order])
            ranked, scores = rrf(lists)
            if rows is not None:  # pages neither retriever put in its top 100 go last, in row order
                seen = set(ranked.tolist())
                tail = np.array([r for r in rows.tolist() if r not in seen], dtype=np.int64)
                ranked = np.concatenate([ranked, tail])
                scores = np.concatenate([scores, np.zeros(len(tail))])
        else:
            s = self._scores(question, method, rows, opts, None)
            order = top_k(s, depth)
            ranked, scores = (order if rows is None else rows[order]), s[order].astype(np.float64)
        if rows is not None and opts.route_statements:
            wanted = question_kinds(question)
            if wanted:
                score_of = dict(zip(ranked.tolist(), scores.tolist(), strict=True))
                ranked = np.array(route(ranked.tolist(), self.kinds, wanted), dtype=np.int64)
                scores = np.array([score_of[r] for r in ranked.tolist()])
        return ranked[:k], scores[:k]

    def ranked_pages(self, question: str, k: int, method: str = "bm25",
                     opts: Options = ALL_PAGES) -> list[tuple[str, int]]:
        rows, _ = self.rank(question, k, method, opts)
        return [self.corpus.key(int(r)) for r in rows]

    def search(self, question: str, k: int = 10, method: str = "bm25",
               opts: Options = ALL_PAGES) -> tuple[list[Hit], dict]:
        t0 = time.perf_counter()
        docs, rows = self.filtered(question, opts)
        picked, scores = self.rank(question, k, method, opts)
        hits = [Hit(int(r), *self.corpus.key(int(r)), float(s)) for r, s in zip(picked, scores, strict=True)]
        info = {
            "method": method,
            "filter": opts.filter,
            "filings": sorted(docs) if docs is not None else None,
            "statements": sorted(question_kinds(question)) if opts.route_statements else [],
            "pages_searched": len(self.corpus) if rows is None else int(len(rows)),
            "ms": round((time.perf_counter() - t0) * 1000, 3),
        }
        return hits, info
