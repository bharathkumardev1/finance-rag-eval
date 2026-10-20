"""Page search: metadata filter, then BM25, MiniLM, or both fused with RRF.

Pages are numbered in one global order (filings sorted by name, pages in filing order), so a
filter result is a set of contiguous row ranges and both retrievers score just those rows.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25, tokenize
from .data import DocMeta, load_doc_meta, load_pages
from .filters import MetadataFilter
from .statements import page_kinds, question_kinds

METHODS = ("bm25", "dense", "hybrid")
RRF_K = 60
RRF_DEPTH = 100


@dataclass(frozen=True)
class Options:
    """What runs around the retriever. The defaults are the full system."""

    filter: str = "company_year"      # none | company | company_year
    quarters: bool = True             # company_year also narrows to a quarter's filing
    drop_company_terms: bool = True   # BM25 query without the company's name once filtered
    route_statements: bool = True     # statement questions go to statement pages first

    def key(self) -> str:
        parts = [self.filter]
        if self.filter == "company_year" and self.quarters:
            parts.append("quarter")
        if self.filter != "none" and self.drop_company_terms:
            parts.append("drop")
        if self.filter != "none" and self.route_statements:
            parts.append("route")
        return "+".join(parts)


FULL = Options()


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
    """Reciprocal rank fusion. Returns fused rows and scores, best first."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking.tolist()):
            fused[row] = fused.get(row, 0.0) + 1.0 / (k + rank + 1)
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

    def rows_for(self, docs: set[str]) -> np.ndarray:
        ranges = [np.arange(*self.offsets[d]) for d in sorted(docs) if d in self.offsets]
        return np.concatenate(ranges) if ranges else np.empty(0, dtype=np.int64)

    def digest(self) -> str:
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
    """BM25 over every page, cached under ``cache_dir`` and keyed by a hash of the page text."""
    prefix = Path(cache_dir) / f"bm25-{corpus.digest()}"
    if Path(f"{prefix}.npz").exists() and Path(f"{prefix}.json").exists():
        return BM25.load(prefix)
    index = BM25(tokenize(t) for t in corpus.texts)
    try:
        prefix.parent.mkdir(parents=True, exist_ok=True)
        index.save(prefix)
    except OSError:  # read-only data volume: keep the index in memory only
        pass
    return index


class Searcher:
    def __init__(self, corpus: Corpus, bm25: BM25, flt: MetadataFilter,
                 dense=None, encoder=None):
        self.corpus = corpus
        self.bm25 = bm25
        self.filter = flt
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

    def filtered(self, question: str, level: str, quarters: bool = True
                 ) -> tuple[set[str] | None, np.ndarray | None]:
        """Filings the filter allows and their page rows; (None, None) means the whole corpus."""
        docs = self.filter.allowed_docs(question, level, quarters)
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
            raise RuntimeError("dense index not loaded")
        if qvec is None:
            qvec = self.encoder.encode([question])[0]
        return self.dense.scores(qvec, rows)

    def rank(self, question: str, method: str, rows: np.ndarray | None, k: int,
             qvec: np.ndarray | None = None, opts: Options = FULL):
        """Top-k page rows and scores over ``rows`` (None = the whole corpus)."""
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
        # With a filter the candidate set is a few hundred pages, so rank all of them; that
        # way statement routing can promote a page no matter how low the retriever put it.
        depth = k if rows is None else len(rows)
        if method == "hybrid":
            if qvec is None and self.encoder is not None:
                qvec = self.encoder.encode([question])[0]
            lists = []
            for m in ("bm25", "dense"):
                s = self._scores(question, m, rows, opts, qvec)
                order = top_k(s, RRF_DEPTH)
                lists.append(order if rows is None else rows[order])
            ranked, scores = rrf(lists)
            if rows is not None:  # pages neither retriever put in its top 100 go last, in row order
                seen = set(ranked.tolist())
                tail = np.array([r for r in rows.tolist() if r not in seen], dtype=np.int64)
                ranked = np.concatenate([ranked, tail])
                scores = np.concatenate([scores, np.zeros(len(tail))])
        else:
            s = self._scores(question, method, rows, opts, qvec)
            order = top_k(s, depth)
            ranked, scores = (order if rows is None else rows[order]), s[order]
        if rows is not None and opts.route_statements:
            wanted = question_kinds(question)
            if wanted:
                pos = {r: i for i, r in enumerate(ranked.tolist())}
                promoted = sorted((r for r in ranked.tolist() if self.kinds[r] & wanted),
                                  key=pos.__getitem__)
                if promoted:
                    rest = [r for r in ranked.tolist() if not self.kinds[r] & wanted]
                    new = np.array(promoted + rest, dtype=np.int64)
                    score_of = dict(zip(ranked.tolist(), scores.tolist(), strict=True))
                    ranked, scores = new, np.array([score_of[r] for r in new.tolist()])
        return ranked[:k], np.asarray(scores[:k], dtype=np.float64)

    def search(self, question: str, method: str = "hybrid", k: int = 10,
               opts: Options = FULL) -> tuple[list[Hit], dict]:
        t0 = time.perf_counter()
        docs, rows = self.filtered(question, opts.filter, opts.quarters)
        picked, scores = self.rank(question, method, rows, k, opts=opts)
        hits = [Hit(int(r), self.corpus.row_doc[r], self.corpus.row_page[r], float(s))
                for r, s in zip(picked, scores, strict=True)]
        info = {
            "method": method,
            "filter": opts.filter,
            "filings": sorted(docs) if docs is not None else None,
            "statements": sorted(question_kinds(question)) if opts.route_statements else [],
            "pages_searched": len(self.corpus) if rows is None else int(len(rows)),
            "ms": round((time.perf_counter() - t0) * 1000, 3),
        }
        return hits, info


def load_aliases(path: Path) -> dict[str, list[str]]:
    import yaml

    raw = yaml.safe_load(Path(path).read_text()) or {}
    return {company: [str(a) for a in names] for company, names in raw.items()}


def write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n")
