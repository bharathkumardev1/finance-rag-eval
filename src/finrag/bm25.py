"""BM25 over filing pages.

I wrote this out instead of calling bm25s so that a JavaScript port can run the same formula
on the same tokens and rank pages identically. It is the Lucene variant, with bm25s's
defaults:

    idf(t)    = ln(1 + (N - df + 0.5) / (df + 0.5))
    score(p)  = sum over query tokens t of  idf(t) * tf / (tf + k1 * (1 - b + b * len(p) / avg_len))

A token is a lowercased run of two or more word characters, minus Lucene's 33 English
stopwords, which is what ``bm25s.tokenize(stopwords="en")`` produces. Repeated query tokens
count once per occurrence, as in bm25s. ``tests/test_bm25.py`` checks the scores against bm25s.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scipy import sparse

TOKEN = re.compile(r"(?u)\b\w\w+\b")
STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the their "
    "then there these they this to was will with".split()
)
K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    """Term-major sparse index: row t holds the precomputed BM25 weight of term t in each page."""

    def __init__(self, docs: Iterable[list[str]], k1: float = K1, b: float = B):
        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        tfs: list[int] = []
        lengths: list[int] = []
        for i, tokens in enumerate(docs):
            lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                rows.append(vocab.setdefault(term, len(vocab)))
                cols.append(i)
                tfs.append(tf)
        n = len(lengths)
        tf = sparse.csr_matrix(
            (np.asarray(tfs, dtype=np.float64), (rows, cols)), shape=(len(vocab), n)
        )
        tf.sort_indices()
        df = np.diff(tf.indptr)
        dl = np.asarray(lengths, dtype=np.float64)
        avg = dl.mean() if n else 0.0
        self.idf = np.log1p((n - df + 0.5) / (df + 0.5))
        norm = k1 * (1.0 - b + b * dl / avg) if avg else np.full(n, k1)
        term_of_entry = np.repeat(np.arange(len(vocab)), df)
        weights = self.idf[term_of_entry] * tf.data / (tf.data + norm[tf.indices])
        self.weights = sparse.csr_matrix(
            (weights.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape
        )
        self.vocab = vocab
        self.df = df
        self.lengths = dl
        self.avg_len = float(avg)
        self.n_docs = n

    def save(self, path_prefix: Path) -> None:
        sparse.save_npz(f"{path_prefix}.npz", self.weights, compressed=False)
        Path(f"{path_prefix}.json").write_text(json.dumps({
            "vocab": sorted(self.vocab, key=self.vocab.get),
            "lengths": self.lengths.astype(int).tolist(),
        }))

    @classmethod
    def load(cls, path_prefix: Path) -> BM25:
        meta = json.loads(Path(f"{path_prefix}.json").read_text())
        index = cls.__new__(cls)
        index.weights = sparse.load_npz(f"{path_prefix}.npz").tocsr()
        index.vocab = {t: i for i, t in enumerate(meta["vocab"])}
        index.df = np.diff(index.weights.indptr)
        index.lengths = np.asarray(meta["lengths"], dtype=np.float64)
        index.n_docs = len(index.lengths)
        index.avg_len = float(index.lengths.mean()) if index.n_docs else 0.0
        index.idf = np.log1p((index.n_docs - index.df + 0.5) / (index.df + 0.5))
        return index

    def scores(self, tokens: list[str], rows: np.ndarray | None = None) -> np.ndarray:
        """Score every page, or only ``rows`` (sorted page indices) if given."""
        w = self.weights
        if rows is None:
            out = np.zeros(self.n_docs, dtype=np.float32)
            for t in tokens:
                j = self.vocab.get(t)
                if j is not None:
                    lo, hi = w.indptr[j], w.indptr[j + 1]
                    out[w.indices[lo:hi]] += w.data[lo:hi]
            return out
        out = np.zeros(len(rows), dtype=np.float32)
        for t in tokens:
            j = self.vocab.get(t)
            if j is None:
                continue
            lo, hi = w.indptr[j], w.indptr[j + 1]
            idx = w.indices[lo:hi]
            pos = np.searchsorted(rows, idx)
            hit = (pos < len(rows)) & (rows[np.minimum(pos, len(rows) - 1)] == idx)
            out[pos[hit]] += w.data[lo:hi][hit]
        return out
