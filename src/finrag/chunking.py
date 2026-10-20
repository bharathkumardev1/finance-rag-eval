"""Page-aware chunking.

Chunks are windows of words that remember which PDF pages they span. That is what lets a
retrieved chunk be scored against FinanceBench's page-level evidence labels.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    doc_name: str
    text: str
    pages: tuple[int, ...]


def chunk_document(
    doc_name: str, pages: list[str], size: int = 512, overlap: int = 0
) -> list[Chunk]:
    """Split a document into windows of ``size`` words with ``overlap`` words shared.

    Sizes are in whitespace-delimited words, not model tokens, so results do not depend on
    which tokenizer an embedder uses.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")

    words: list[str] = []
    page_of: list[int] = []
    for page_idx, text in enumerate(pages):
        toks = text.split()
        words.extend(toks)
        page_of.extend([page_idx] * len(toks))

    if not words:
        return []

    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        end = min(start + size, len(words))
        chunk_pages = tuple(sorted(set(page_of[start:end])))
        chunks.append(Chunk(doc_name, " ".join(words[start:end]), chunk_pages))
        if end == len(words):
            break
    return chunks


def chunk_corpus(
    docs: dict[str, list[str]], size: int = 512, overlap: int = 0
) -> list[Chunk]:
    out: list[Chunk] = []
    for name in sorted(docs):
        out.extend(chunk_document(name, docs[name], size, overlap))
    return out
