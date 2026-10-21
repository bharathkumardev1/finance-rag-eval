"""BM25 over filing pages, in segments so new filings go in without rebuilding the index.

I wrote this out instead of calling bm25s so that a JavaScript port can run the same formula
on the same tokens and rank pages identically. It is the Lucene variant, with bm25s's
defaults:

    idf(t)    = ln(1 + (N - df + 0.5) / (df + 0.5))
    score(p)  = sum over query tokens t of  idf(t) * tf / (tf + k1 * (1 - b + b * len(p) / avg_len))

A token is a lowercased run of two or more word characters, minus Lucene's 33 English
stopwords, which is what ``bm25s.tokenize(stopwords="en")`` produces. Repeated query tokens
count once per occurrence, as in bm25s. ``tests/test_bm25.py`` checks the scores against bm25s.

The index is a list of segments, as in Lucene. A segment holds raw term counts and page
lengths for the filings added together, and nothing global: N, df and the average length are
summed over segments when a query is scored. Adding filings appends a segment and leaves the
others alone, and the scores come out the same as from an index built in one pass, because
every page gets the same terms in the same order from the same statistics.
``finrag check-incremental`` measures that on the real corpus.

A segment knows its filings by name, not by row number. ``assign_rows`` places them in the
corpus's row order when the index is loaded, so the corpus can grow in any order.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
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


@dataclass
class Segment:
    filings: list[tuple[str, int, str]]   # (name, pages, content hash) in the segment's page order
    tf: sparse.csr_matrix                 # terms x pages, raw counts; term ids from the index vocab
    lengths: np.ndarray                   # tokens per page
    file: str | None = None               # where it is saved, once it is
    rows: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    @property
    def n_terms(self) -> int:
        return self.tf.shape[0]


class BM25:
    def __init__(self, k1: float = K1, b: float = B):
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        self.df = np.zeros(0, dtype=np.int64)
        self.segments: list[Segment] = []
        self.n_docs = 0
        self.total_len = 0

    @classmethod
    def from_pages(cls, pages: Iterable[list[str]], k1: float = K1, b: float = B) -> BM25:
        """One segment over token lists, scored in the order given. Handy for tests."""
        pages = list(pages)
        index = cls(k1, b)
        index.add({"": pages})
        index.assign_rows({"": (0, len(pages))})
        return index

    @property
    def avg_len(self) -> float:
        return self.total_len / self.n_docs if self.n_docs else 0.0

    def filings(self) -> dict[str, tuple[int, str]]:
        return {name: (n, digest) for seg in self.segments for name, n, digest in seg.filings}

    def add(self, filings: dict[str, list[list[str]]], hashes: dict[str, str] | None = None) -> Segment:
        """Append a segment holding these filings (name to per-page token lists)."""
        known = self.filings()
        if dup := sorted(set(filings) & set(known)):
            raise ValueError(f"already indexed: {', '.join(dup[:5])}")
        rows: list[int] = []
        cols: list[int] = []
        counts: list[int] = []
        lengths: list[int] = []
        meta = []
        for name in sorted(filings):
            pages = filings[name]
            meta.append((name, len(pages), (hashes or {}).get(name, "")))
            for tokens in pages:
                col = len(lengths)
                lengths.append(len(tokens))
                for term, n in Counter(tokens).items():
                    rows.append(self.vocab.setdefault(term, len(self.vocab)))
                    cols.append(col)
                    counts.append(n)
        tf = sparse.csr_matrix((np.asarray(counts, dtype=np.int32), (rows, cols)),
                               shape=(len(self.vocab), len(lengths)))
        tf.sort_indices()
        seg = Segment(meta, tf, np.asarray(lengths, dtype=np.int64))
        df = np.zeros(len(self.vocab), dtype=np.int64)
        df[: len(self.df)] = self.df
        df[: seg.n_terms] += np.diff(tf.indptr)
        self.df = df
        self.segments.append(seg)
        self.n_docs += len(lengths)
        self.total_len += int(seg.lengths.sum())
        return seg

    def assign_rows(self, offsets: dict[str, tuple[int, int]]) -> None:
        """Map every indexed page to its row in the corpus, given each filing's [start, end)."""
        for seg in self.segments:
            parts = []
            for name, n, _ in seg.filings:
                start, end = offsets[name]
                if end - start != n:
                    raise ValueError(f"{name}: indexed {n} pages, corpus has {end - start}")
                parts.append(np.arange(start, end, dtype=np.int64))
            seg.rows = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    def idf(self, j: int) -> float:
        """Lucene idf of term id ``j``. The browser export reads the same value, not a recomputation."""
        df = self.df[j]
        return np.log1p((self.n_docs - df + 0.5) / (df + 0.5))

    def page_lengths(self) -> np.ndarray:
        """Tokens per page in corpus row order."""
        out = np.zeros(self.n_docs, dtype=np.int64)
        for seg in self.segments:
            out[seg.rows] = seg.lengths
        return out

    def scores(self, tokens: list[str], rows: np.ndarray | None = None) -> np.ndarray:
        """Score of every page in corpus row order, or of ``rows`` only if given."""
        out = np.zeros(self.n_docs, dtype=np.float64)
        avg = self.avg_len
        for t in tokens:
            j = self.vocab.get(t)
            if j is None:
                continue
            idf = self.idf(j)
            for seg in self.segments:
                if j >= seg.n_terms:
                    continue
                lo, hi = seg.tf.indptr[j], seg.tf.indptr[j + 1]
                if lo == hi:
                    continue
                pages = seg.tf.indices[lo:hi]
                tf = seg.tf.data[lo:hi].astype(np.float64)
                norm = self.k1 * (1.0 - self.b + self.b * seg.lengths[pages] / avg)
                out[seg.rows[pages]] += idf * tf / (tf + norm)
        return out if rows is None else out[rows]

    def save(self, directory: Path) -> None:
        """Write segments that aren't on disk yet, then the vocabulary and the manifest."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for i, seg in enumerate(self.segments):
            if seg.file is None:
                seg.file = f"segment-{i:04d}.npz"
                tmp = directory / f"{seg.file}.part.npz"
                np.savez(tmp, data=seg.tf.data, indices=seg.tf.indices, indptr=seg.tf.indptr,
                         shape=np.asarray(seg.tf.shape), lengths=seg.lengths)
                tmp.replace(directory / seg.file)
        (directory / "vocab.json").write_text(json.dumps(sorted(self.vocab, key=self.vocab.get)))
        manifest = {"k1": self.k1, "b": self.b, "segments": [
            {"file": seg.file, "filings": [list(f) for f in seg.filings]} for seg in self.segments]}
        tmp = directory / "manifest.json.part"
        tmp.write_text(json.dumps(manifest, indent=1))
        tmp.replace(directory / "manifest.json")  # last, so a crash leaves the old manifest

    @classmethod
    def load(cls, directory: Path) -> BM25:
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        index = cls(manifest["k1"], manifest["b"])
        terms = json.loads((directory / "vocab.json").read_text())
        index.vocab = {t: i for i, t in enumerate(terms)}
        index.df = np.zeros(len(terms), dtype=np.int64)
        for entry in manifest["segments"]:
            z = np.load(directory / entry["file"])
            tf = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
            seg = Segment([tuple(f) for f in entry["filings"]], tf, z["lengths"], entry["file"])
            index.df[: seg.n_terms] += np.diff(tf.indptr)
            index.segments.append(seg)
            index.n_docs += len(seg.lengths)
            index.total_len += int(seg.lengths.sum())
        return index
