"""Dense retrieval with sentence-transformers/all-MiniLM-L6-v2, run through ONNX Runtime.

No PyTorch. The ONNX file is pinned by SHA-256 to the official Hugging Face upload, so a
mirror can't swap in different weights. Embeddings match sentence-transformers: WordPiece
tokens, mean pooling over the attention mask, L2 normalization.

The model was trained on 256-token inputs, so a page is split into 254-token windows (plus
[CLS] and [SEP]) that never cross a page boundary, and a page scores as its best window
(MaxP). Queries are short enough to fit in one window.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_SHA256 = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
HF_BASE = f"https://huggingface.co/{MODEL_ID}/resolve/main"
MAX_TOKENS = 256
DIM = 384


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_model(dest: Path, base: str = HF_BASE) -> Path:
    """Fetch ``onnx/model.onnx`` and ``vocab.txt``; refuse a model that isn't the official one."""
    dest = Path(dest)
    (dest / "onnx").mkdir(parents=True, exist_ok=True)
    for name in ("onnx/model.onnx", "vocab.txt"):
        target = dest / name
        if not target.exists():
            tmp = target.with_suffix(target.suffix + ".part")
            urllib.request.urlretrieve(f"{base}/{name}", tmp)
            tmp.rename(target)
    got = file_sha256(dest / "onnx" / "model.onnx")
    if got != MODEL_SHA256:
        raise RuntimeError(f"model.onnx sha256 {got} does not match the official {MODEL_SHA256}")
    return dest


class MiniLM:
    name = "minilm-l6-v2"
    dim = DIM
    sha256 = MODEL_SHA256

    def __init__(self, model_dir: Path, threads: int = 1, batch_size: int = 64,
                 onnx_file: str = "onnx/model.onnx", verify: bool = True):
        import onnxruntime as ort
        from tokenizers import BertWordPieceTokenizer

        model_dir = Path(model_dir)
        path = model_dir / onnx_file
        if verify and file_sha256(path) != MODEL_SHA256:
            raise RuntimeError(f"{path} is not the official {MODEL_ID} export")
        self.batch_size = batch_size
        self._tok = BertWordPieceTokenizer(str(model_dir / "vocab.txt"), lowercase=True)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self._cls = self._tok.token_to_id("[CLS]")
        self._sep = self._tok.token_to_id("[SEP]")

    def windows(self, text: str, max_windows: int | None = None) -> list[list[int]]:
        ids = self._tok.encode(text, add_special_tokens=False).ids
        body = MAX_TOKENS - 2
        wins = [ids[i : i + body] for i in range(0, max(len(ids), 1), body)]
        if max_windows is not None:
            wins = wins[:max_windows]
        return [[self._cls, *w, self._sep] for w in wins]

    def embed_ids(self, seqs: list[list[int]]) -> np.ndarray:
        out = np.zeros((len(seqs), self.dim), dtype=np.float32)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))  # less padding per batch
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            width = max(len(seqs[i]) for i in idx)
            ids = np.zeros((len(idx), width), dtype=np.int64)
            mask = np.zeros((len(idx), width), dtype=np.int64)
            for row, i in enumerate(idx):
                ids[row, : len(seqs[i])] = seqs[i]
                mask[row, : len(seqs[i])] = 1
            hidden = self._sess.run(
                None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            )[0]
            m = mask[..., None].astype(np.float32)
            out[idx] = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        out /= np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)
        return out

    def encode(self, texts: list[str]) -> np.ndarray:
        """One vector per text from its first window. Used for queries."""
        return self.embed_ids([self.windows(t, max_windows=1)[0] for t in texts])

    def encode_pages(self, pages: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Window vectors for every page, and the page index each window came from."""
        seqs, owner = [], []
        for p, text in enumerate(pages):
            for w in self.windows(text):
                seqs.append(w)
                owner.append(p)
        return self.embed_ids(seqs), np.array(owner, dtype=np.int32)


def pages_digest(pages: list[str]) -> str:
    return hashlib.sha256("\f".join(pages).encode("utf8")).hexdigest()[:16]


def embed_corpus(model, data_dir: Path, out_dir: Path) -> dict:
    """Embed every page of every cached filing into ``out_dir/<doc>.npz``.

    Incremental: a filing is skipped when its file exists and was built from the same page
    text (by hash) with the same model, so an interrupted run picks up where it stopped and a
    re-extracted filing gets re-embedded. Vectors are stored as float16.
    """
    from .data import load_pages

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for doc, pages in load_pages(data_dir).items():
        digest = pages_digest(pages)
        target = out_dir / f"{doc}.npz"
        if target.exists():
            try:
                meta = json.loads(str(np.load(target, allow_pickle=False)["meta"]))
            except Exception:  # a file cut short by an interrupted run: embed it again
                meta = {}
            if meta.get("pages_sha") == digest and meta.get("model_sha256") == model.sha256:
                skipped += 1
                continue
        t0 = time.perf_counter()
        vecs, owner = model.encode_pages(pages)
        meta = json.dumps({"model_sha256": model.sha256, "pages_sha": digest, "n_pages": len(pages)})
        tmp = out_dir / f"{doc}.tmp.npz"
        np.savez(tmp, vectors=vecs.astype(np.float16), page=owner, meta=np.array(meta))
        tmp.rename(target)
        done += 1
        print(f"{doc}: {len(pages)} pages, {len(owner)} windows, {time.perf_counter() - t0:.1f}s",
              flush=True)
    return {"embedded": done, "skipped": skipped}


class DenseIndex:
    """All window vectors in page order, so a page's windows are one contiguous slice."""

    def __init__(self, vectors: np.ndarray, window_page: np.ndarray, n_pages: int):
        order = np.argsort(window_page, kind="stable")
        self.vectors = np.ascontiguousarray(vectors[order], dtype=np.float32)
        self.window_page = window_page[order]
        counts = np.bincount(self.window_page, minlength=n_pages)
        if (counts == 0).any():
            raise ValueError(f"{int((counts == 0).sum())} pages have no window vectors")
        self.page_start = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        self.page_end = self.page_start + counts
        self.n_pages = n_pages

    @classmethod
    def load(cls, emb_dir: Path, doc_offsets: dict[str, tuple[int, int]]) -> DenseIndex:
        """``doc_offsets`` maps each filing to its [start, end) range of global page rows."""
        vecs, owners = [], []
        n_pages = max(end for _, end in doc_offsets.values())
        for doc, (start, end) in doc_offsets.items():
            path = Path(emb_dir) / f"{doc}.npz"
            if not path.exists():
                raise FileNotFoundError(f"no embeddings for {doc}; run `finrag embed`")
            z = np.load(path, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            if meta["n_pages"] != end - start:
                raise ValueError(f"{doc}: embedded {meta['n_pages']} pages, corpus has {end - start}")
            vecs.append(z["vectors"])
            owners.append(z["page"].astype(np.int64) + start)
        return cls(np.concatenate(vecs), np.concatenate(owners), n_pages)

    def scores(self, qvec: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        """MaxP score for every page, or for ``rows`` (sorted page indices) if given."""
        if rows is None:
            sims = self.vectors @ qvec
            return np.maximum.reduceat(sims, self.page_start).astype(np.float32)
        starts, ends = self.page_start[rows], self.page_end[rows]
        lengths = ends - starts
        idx = np.repeat(starts - np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths)
        idx += np.arange(lengths.sum())
        sims = self.vectors[idx] @ qvec
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        return np.maximum.reduceat(sims, offsets).astype(np.float32)
