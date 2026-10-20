"""Sparse, dense, and hybrid retrievers with optional metadata masks.

Every retriever returns chunk indices ranked best first. A ``mask`` is a boolean array over
all chunks; when given, only chunks where it is True can be returned. Dense search over a
mask is exact (a dot product over the allowed rows), which is how a metadata pre-filter
behaves in a vector store once it narrows the corpus to a few filings.
"""

from __future__ import annotations

import numpy as np

from .embedders import Embedder


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    part = np.argpartition(-scores, k - 1)[:k]
    order = part[np.argsort(-scores[part], kind="stable")]
    return order[np.isfinite(scores[order])]


class BM25Retriever:
    name = "bm25"

    def __init__(self, texts: list[str]):
        import bm25s

        self._bm25s = bm25s
        self._model = bm25s.BM25()
        self._model.index(bm25s.tokenize(texts, stopwords="en", show_progress=False),
                          show_progress=False)
        self.size = len(texts)

    def scores(self, query: str) -> np.ndarray:
        tokens = self._bm25s.tokenize(
            [query], stopwords="en", show_progress=False, return_ids=False
        )[0]
        known = [t for t in tokens if t in self._model.vocab_dict]
        if not known:
            return np.zeros(self.size, dtype=np.float32)
        return np.asarray(self._model.get_scores(known), dtype=np.float32)

    def search(self, query: str, k: int, mask: np.ndarray | None = None) -> list[int]:
        s = self.scores(query)
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        return _top_k(s, k).tolist()


def build_faiss_index(vectors: np.ndarray, spec: str = "flat", **params):
    """Build an inner-product FAISS index. Vectors must be L2-normalized (cosine similarity).

    ``flat`` is exact. ``hnsw`` takes ``M`` and ``ef_construction``. ``ivf`` takes ``nlist``.
    """
    import faiss

    d = vectors.shape[1]
    if spec == "flat":
        index = faiss.IndexFlatIP(d)
    elif spec == "hnsw":
        index = faiss.IndexHNSWFlat(d, int(params.get("M", 32)), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = int(params.get("ef_construction", 200))
    elif spec == "ivf":
        nlist = int(params.get("nlist", max(16, int(4 * np.sqrt(len(vectors))))))
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
    else:
        raise ValueError(f"Unknown index spec: {spec}")
    index.add(vectors)
    return index


def set_search_param(index, spec: str, value: int) -> None:
    if spec == "hnsw":
        index.hnsw.efSearch = int(value)
    elif spec == "ivf":
        index.nprobe = int(value)


class DenseRetriever:
    name = "dense"

    def __init__(self, embedder: Embedder, vectors: np.ndarray, index=None):
        self.embedder = embedder
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index = index if index is not None else build_faiss_index(self.vectors, "flat")

    def scores_for(self, qvec: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return self.vectors[idx] @ qvec

    def search_vector(self, qvec: np.ndarray, k: int, mask: np.ndarray | None = None) -> list[int]:
        if mask is None:
            _, ids = self.index.search(qvec[None, :].astype(np.float32), k)
            return [int(i) for i in ids[0] if i >= 0]
        allowed = np.flatnonzero(mask)
        if allowed.size == 0:
            return []
        order = _top_k(self.scores_for(qvec, allowed), k)
        return allowed[order].tolist()

    def search(self, query: str, k: int, mask: np.ndarray | None = None) -> list[int]:
        qvec = self.embedder.encode([query])[0]
        return self.search_vector(qvec, k, mask)


class HybridRetriever:
    """Reciprocal rank fusion (Cormack et al., 2009) of BM25 and dense rankings."""

    name = "hybrid"

    def __init__(self, sparse: BM25Retriever, dense: DenseRetriever, depth: int = 100,
                 k_rrf: int = 60):
        self.sparse = sparse
        self.dense = dense
        self.depth = depth
        self.k_rrf = k_rrf

    def search(self, query: str, k: int, mask: np.ndarray | None = None) -> list[int]:
        fused: dict[int, float] = {}
        for ranking in (
            self.sparse.search(query, self.depth, mask),
            self.dense.search(query, self.depth, mask),
        ):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.k_rrf + rank + 1)
        return [i for i, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]
