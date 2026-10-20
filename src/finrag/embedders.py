"""Embedders behind one interface: ``fit`` on the corpus (if needed), ``encode`` to unit vectors.

Specs accepted by :func:`make_embedder`:

* ``lsa-<dim>``: TF-IDF plus truncated SVD fitted on the corpus. Runs offline with no model
  download, so it is the default baseline and what CI uses.
* ``hashing-<dim>``: feature hashing with no fitting. Used in tests.
* ``st:<model>``: any sentence-transformers model, for example ``st:BAAI/bge-small-en-v1.5``.
* ``openai:<model>``: OpenAI embeddings, for example ``openai:text-embedding-3-small``.
"""

from __future__ import annotations

import os
import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"(?u)\b[\w$%.-]*\w\b")


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class Embedder(Protocol):
    name: str
    dim: int

    def fit(self, texts: list[str]) -> None: ...

    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self.name = f"hashing-{dim}"
        self._vec = HashingVectorizer(
            n_features=dim, alternate_sign=False, norm=None, token_pattern=_TOKEN.pattern
        )

    def fit(self, texts: list[str]) -> None:
        return None

    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        m = self._vec.transform(texts)
        m.data = np.log1p(m.data)
        return _normalize(m.toarray())


class LSAEmbedder:
    """Latent semantic analysis: a dense baseline learned from the corpus itself."""

    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim = dim
        self.name = f"lsa-{dim}"
        self.seed = seed
        self._tfidf = None
        self._svd = None

    def fit(self, texts: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._tfidf = TfidfVectorizer(
            sublinear_tf=True,
            min_df=2,
            max_df=0.5,
            token_pattern=_TOKEN.pattern,
            dtype=np.float32,
        )
        x = self._tfidf.fit_transform(texts)
        self._svd = TruncatedSVD(n_components=self.dim, random_state=self.seed)
        self._svd.fit(x)

    def encode(self, texts: list[str], batch_size: int = 4096) -> np.ndarray:
        if self._tfidf is None or self._svd is None:
            raise RuntimeError("LSAEmbedder.fit must be called before encode")
        parts = [
            self._svd.transform(self._tfidf.transform(texts[i : i + batch_size]))
            for i in range(0, len(texts), batch_size)
        ]
        return _normalize(np.vstack(parts) if parts else np.zeros((0, self.dim)))


class SentenceTransformerEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self._model = SentenceTransformer(model)
        self.name = f"st:{model}"
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def fit(self, texts: list[str]) -> None:
        return None

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vecs = self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return _normalize(vecs)


class OpenAIEmbedder:
    _DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI  # optional dependency

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY to use OpenAI embeddings")
        self._client = OpenAI()
        self._model = model
        self.name = f"openai:{model}"
        self.dim = self._DIMS.get(model, 1536)

    def fit(self, texts: list[str]) -> None:
        return None

    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            out.extend(d.embedding for d in resp.data)
        return _normalize(np.array(out))


def make_embedder(spec: str) -> Embedder:
    if spec.startswith("lsa-"):
        return LSAEmbedder(int(spec.split("-", 1)[1]))
    if spec.startswith("hashing-"):
        return HashingEmbedder(int(spec.split("-", 1)[1]))
    if spec.startswith("st:"):
        return SentenceTransformerEmbedder(spec[3:])
    if spec.startswith("openai:"):
        return OpenAIEmbedder(spec[7:])
    raise ValueError(f"Unknown embedder spec: {spec}")
