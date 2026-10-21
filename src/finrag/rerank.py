"""Reranker study: BAAI/bge-reranker-base over the top 30 pages of a retrieval system.

A cross-encoder reads the question and a page together, so it can match "fixed asset
turnover" to a page that never uses the phrase. That costs a full transformer pass per
(question, page) pair, so it only reorders the top 30 of a cheaper ranking. Pages 31 and
below keep their order after them.

The rankings come from results/eval.json, so this step doesn't rerun retrieval or need the
MiniLM embeddings. The ONNX weights are the official BAAI upload at a pinned revision,
checked by SHA-256 (1.1 GB). A question and page are truncated to the model's 512 tokens
together, cutting the page, so the end of a long page goes unread.

Whether reranking helps is decided on dev; test is reported next to it.
"""

from __future__ import annotations

import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .data import Question, sha256_file
from .evaluate import compare
from .metrics import bootstrap_ci, metrics, per_question, score_ranking

MODEL_ID = "BAAI/bge-reranker-base"
REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"
ONNX_SHA256 = "15b9a8c3da82eddf263df571281166e00e9308fe19d077084b642ebfcaf06d2b"
HF_BASE = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}"
FILES = ("onnx/model.onnx", "tokenizer.json")
MAX_TOKENS = 512
TOP = 30


def download_reranker(dest: Path, base: str = HF_BASE) -> Path:
    dest = Path(dest)
    for name in FILES:
        target = dest / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            urllib.request.urlretrieve(f"{base}/{name}", tmp)
            tmp.replace(target)
    got = sha256_file(dest / "onnx" / "model.onnx")
    if got != ONNX_SHA256:
        raise RuntimeError(f"model.onnx has sha256 {got}, not the official {ONNX_SHA256}")
    return dest


class CrossEncoder:
    def __init__(self, model_dir: Path, threads: int = 4, batch_size: int = 8):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        path = model_dir / "onnx" / "model.onnx"
        if sha256_file(path) != ONNX_SHA256:
            raise RuntimeError(f"{path} is not the official {MODEL_ID} export")
        self.tokenizer_sha256 = sha256_file(model_dir / "tokenizer.json")
        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tok.enable_truncation(MAX_TOKENS, strategy="only_second")
        self._tok.enable_padding(pad_id=self._tok.token_to_id("<pad>"), pad_token="<pad>")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self._sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self.batch_size = batch_size

    def score(self, question: str, passages: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(passages), self.batch_size):
            encs = self._tok.encode_batch([(question, p) for p in passages[i : i + self.batch_size]])
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": np.array([e.attention_mask for e in encs],
                                                                 dtype=np.int64)}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            out.append(self._sess.run(None, feed)[0].reshape(-1))
        return np.concatenate(out) if out else np.empty(0)


def rerank(ranked: list[list], scores: np.ndarray) -> list[list]:
    """The first len(scores) pages by score, ties kept in their old order; the rest after."""
    head = ranked[: len(scores)]
    order = sorted(range(len(head)), key=lambda i: -scores[i])
    return [head[i] for i in order] + ranked[len(scores):]


def study(evaluation: dict, questions: list[Question], page_text: Callable[[str, int], str],
          score: Callable[[str, list[str]], np.ndarray], config: str) -> dict:
    """Rerank ``config``'s top 30 for every question; score both rankings the same way."""
    reranked_name = f"{config}+rerank"
    per_q: dict[str, dict] = {}
    seconds = []
    for q in questions:
        x = evaluation["questions"][q.qid]
        base = x["runs"][config]["top"][:TOP]
        t0 = time.perf_counter()
        s = score(q.text, [page_text(d, p) for d, p in base])
        seconds.append(time.perf_counter() - t0)
        new = rerank(base, s)
        per_q[q.qid] = {"split": x["split"], "type": x["type"], "runs": {
            config: score_ranking([tuple(k) for k in base], q),
            reranked_name: score_ranking([tuple(k) for k in new], q)},
            "top_after": new[:10]}
    table = []
    for name in (config, reranked_name):
        for part in ("dev", "test"):
            recs = [v["runs"][name] for v in per_q.values() if v["split"] == part]
            if not recs:
                continue
            row = {"config": name, "split": part, "n": len(recs), **metrics(recs)}
            for m in ("hit@10", "mrr@10"):
                row[f"{m}_ci95"] = list(bootstrap_ci(per_question(recs, m)))
            table.append(row)
    res = {"base": config, "top": TOP, "table": table, "questions": per_q,
           "seconds_per_question": {"p50": float(np.percentile(seconds, 50)),
                                    "p95": float(np.percentile(seconds, 95))}}
    res["comparisons"] = [compare(res, reranked_name, config, part, m)
                          for part in ("dev", "test") for m in ("hit@1", "hit@10", "mrr@10")
                          if any(v["split"] == part for v in per_q.values())]
    return res
