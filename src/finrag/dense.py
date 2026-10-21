"""Dense retrieval with sentence-transformers/all-MiniLM-L6-v2, run through ONNX Runtime.

No PyTorch. The ONNX weights are pinned by SHA-256 to the official Hugging Face upload, so a
mirror can't swap in different weights. A text becomes a vector the way sentence-transformers
does it for this model: lowercased WordPiece tokens between [CLS] and [SEP], mean pooling over
the attention mask, L2 normalization.

The model was trained on inputs of up to 256 tokens, so a page is cut into windows of 254
tokens plus [CLS] and [SEP]. Windows never cross a page boundary, and a page scores as its
best window (MaxP). Questions fit in one window.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from .data import pages_digest, sha256_file

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_SHA256 = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
HF_BASE = f"https://huggingface.co/{MODEL_ID}/resolve/main"
MODEL_FILES = ("onnx/model.onnx", "tokenizer.json")
MAX_TOKENS = 256
DIM = 384


def download_model(dest: Path, base: str = HF_BASE) -> Path:
    """Fetch the ONNX export and tokenizer; refuse weights that aren't the official ones."""
    dest = Path(dest)
    for name in MODEL_FILES:
        target = dest / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            urllib.request.urlretrieve(f"{base}/{name}", tmp)
            tmp.replace(target)
    got = sha256_file(dest / "onnx" / "model.onnx")
    if got != MODEL_SHA256:
        raise RuntimeError(f"model.onnx has sha256 {got}, not the official {MODEL_SHA256}")
    return dest


class MiniLM:
    name = "all-MiniLM-L6-v2"
    dim = DIM
    sha256 = MODEL_SHA256

    def __init__(self, model_dir: Path, threads: int = 1, batch_size: int = 64):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        path = model_dir / "onnx" / "model.onnx"
        if sha256_file(path) != MODEL_SHA256:
            raise RuntimeError(f"{path} is not the official {MODEL_ID} export")
        self.tokenizer_sha256 = sha256_file(model_dir / "tokenizer.json")
        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        # The published tokenizer.json truncates and pads to 128; windowing is done here instead.
        self._tok.no_truncation()
        self._tok.no_padding()
        self._cls = self._tok.token_to_id("[CLS]")
        self._sep = self._tok.token_to_id("[SEP]")
        self.batch_size = batch_size
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

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
        """One vector per text, from its first window. Used for questions."""
        return self.embed_ids([self.windows(t, max_windows=1)[0] for t in texts])

    def encode_pages(self, pages: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """A vector per window of every page, and the page each window came from."""
        seqs, owner = [], []
        for p, text in enumerate(pages):
            for w in self.windows(text):
                seqs.append(w)
                owner.append(p)
        return self.embed_ids(seqs), np.array(owner, dtype=np.int32)


def embed_corpus(model, data_dir: Path, out_dir: Path, git_sha: str | None = None) -> dict:
    """Embed every page of every filing into ``out_dir/<filing>.npz``, as float16.

    A filing is skipped when its file was built from the same page text with the same
    model, so an interrupted run picks up where it stopped and a re-extracted filing is
    embedded again.
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
            except Exception:  # cut short by an interrupted run: embed it again
                meta = {}
            if meta.get("pages_sha") == digest and meta.get("model_sha256") == model.sha256:
                skipped += 1
                continue
        t0 = time.perf_counter()
        vecs, owner = model.encode_pages(pages)
        meta = json.dumps({"model_sha256": model.sha256, "pages_sha": digest,
                           "n_pages": len(pages), "git_sha": git_sha})
        tmp = out_dir / f"{doc}.part.npz"
        np.savez(tmp, vectors=vecs.astype(np.float16), page=owner, meta=np.array(meta))
        tmp.replace(target)
        done += 1
        print(f"{doc}: {len(pages)} pages, {len(owner)} windows, {time.perf_counter() - t0:.1f}s",
              flush=True)
    return {"embedded": done, "skipped": skipped}


class DenseIndex:
    """All window vectors in page order, so each page's windows are one contiguous slice."""

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

    def __len__(self) -> int:
        return len(self.window_page)

    def scores(self, qvec: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        """MaxP score of every page, or of ``rows`` (sorted page indices) if given."""
        if rows is None:
            sims = self.vectors @ qvec
            return np.maximum.reduceat(sims, self.page_start).astype(np.float32)
        if len(rows) == 0:
            return np.empty(0, dtype=np.float32)
        starts, ends = self.page_start[rows], self.page_end[rows]
        lengths = ends - starts
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        idx = np.repeat(starts - offsets, lengths) + np.arange(lengths.sum())
        sims = self.vectors[idx] @ qvec
        return np.maximum.reduceat(sims, offsets).astype(np.float32)
