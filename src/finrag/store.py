"""Incremental vector index keyed by document content hash.

Re-embedding a whole corpus for every new filing does not scale. ``IncrementalIndex`` keeps a
manifest of each document's SHA-256 and chunk ids, so an upsert embeds only new or changed
documents, removes a changed document's old chunks, and leaves everything else in place.
Every chunk id maps back to its source document and pages.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_document
from .embedders import Embedder


def content_hash(pages: list[str]) -> str:
    return hashlib.sha256("\f".join(pages).encode("utf8")).hexdigest()


class IncrementalIndex:
    def __init__(self, embedder: Embedder, chunk_size: int = 512, overlap: int = 0):
        import faiss

        self._faiss = faiss
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(embedder.dim))
        self.manifest: dict[str, dict] = {}
        self.chunks: dict[int, Chunk] = {}
        self._next_id = 0

    def __len__(self) -> int:
        return self.index.ntotal

    def upsert(self, doc_name: str, pages: list[str]) -> str:
        """Add or refresh one document. Returns ``added``, ``replaced``, or ``unchanged``."""
        digest = content_hash(pages)
        prior = self.manifest.get(doc_name)
        if prior and prior["sha256"] == digest:
            return "unchanged"
        if prior:
            self.remove(doc_name)
        new = chunk_document(doc_name, pages, self.chunk_size, self.overlap)
        ids = np.arange(self._next_id, self._next_id + len(new), dtype=np.int64)
        if len(new):
            vecs = self.embedder.encode([c.text for c in new])
            self.index.add_with_ids(vecs, ids)
        self._next_id += len(new)
        for i, c in zip(ids.tolist(), new, strict=True):
            self.chunks[i] = c
        self.manifest[doc_name] = {"sha256": digest, "ids": ids.tolist()}
        return "replaced" if prior else "added"

    def remove(self, doc_name: str) -> int:
        entry = self.manifest.pop(doc_name, None)
        if not entry:
            return 0
        ids = np.array(entry["ids"], dtype=np.int64)
        if ids.size:
            self.index.remove_ids(ids)
        for i in entry["ids"]:
            self.chunks.pop(i, None)
        return int(ids.size)

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        qvec = self.embedder.encode([query])
        scores, ids = self.index.search(qvec, min(k, self.index.ntotal))
        return [(self.chunks[int(i)], float(s)) for s, i in zip(scores[0], ids[0], strict=True)
                if i >= 0]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(path / "index.faiss"))
        state = {
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "next_id": self._next_id,
            "manifest": self.manifest,
            "chunks": {str(i): [c.doc_name, c.text, list(c.pages)] for i, c in self.chunks.items()},
        }
        (path / "state.json").write_text(json.dumps(state), encoding="utf8")

    @classmethod
    def load(cls, path: Path, embedder: Embedder) -> IncrementalIndex:
        path = Path(path)
        state = json.loads((path / "state.json").read_text(encoding="utf8"))
        obj = cls(embedder, state["chunk_size"], state["overlap"])
        obj.index = obj._faiss.read_index(str(path / "index.faiss"))
        obj.manifest = state["manifest"]
        obj._next_id = state["next_id"]
        obj.chunks = {int(i): Chunk(d, t, tuple(p)) for i, (d, t, p) in state["chunks"].items()}
        return obj
