"""Reranker study: BAAI/bge-reranker-base over the top 30 pages of a retrieval system.

A cross-encoder reads the question and a page together, so it can match "fixed asset
turnover" to a page that never uses the phrase. That costs a full transformer pass per
(question, page) pair, so it only reorders the top 30 of a cheaper ranking. Pages 31 and
below keep their order after them.

The rankings come from results/eval.json, so this step doesn't rerun retrieval or need the
MiniLM embeddings. The ONNX weights are the official BAAI upload at a pinned revision,
checked by SHA-256 (1.1 GB). The model reads 512 tokens, question and page together. With
``windows=1`` the page is cut there and the end of a long page goes unread; a median page runs
about 800 tokens. With ``windows=3`` the page is cut into up to three windows that each get the
question, and a page scores as its best window, the way a page scores as its best MiniLM window.

Several systems can be reranked in one run, sharing the scores for pages they have in common,
and each reranked system is compared with every system studied. Whether reranking helps is
decided on dev; test is reported next to it.
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
    def __init__(self, model_dir: Path, threads: int = 4, batch_size: int = 8, windows: int = 1):
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
        self._plain = Tokenizer.from_file(str(model_dir / "tokenizer.json"))  # no truncation, no padding
        self.windows = windows
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self._sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self.batch_size = batch_size

    def score(self, question: str, passages: list[str]) -> np.ndarray:
        """One score per passage: its best window's score (the only window when ``windows`` is 1)."""
        if self.windows == 1:
            return self._score_pairs(question, passages)
        room = MAX_TOKENS - SPECIAL_TOKENS - len(self._plain.encode(question, add_special_tokens=False).ids)
        pieces, owner = [], []
        for i, text in enumerate(passages):
            parts = split_windows(text, self._plain.encode(text, add_special_tokens=False).offsets, room,
                                  self.windows)
            pieces += parts
            owner += [i] * len(parts)
        return best_per_owner(self._score_pairs(question, pieces), owner, len(passages))

    def _score_pairs(self, question: str, passages: list[str]) -> np.ndarray:
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


SPECIAL_TOKENS = 4  # <s> question </s></s> page </s>


def split_windows(text: str, offsets: list[tuple[int, int]], size: int, limit: int) -> list[str]:
    """Up to ``limit`` consecutive pieces of ``text``, each at most ``size`` tokens, cut at token
    edges. ``offsets`` are the (start, end) character positions of the text's tokens. Text past
    the last window is dropped, and a text that fits comes back whole."""
    if size < 1:
        raise ValueError("no room for page tokens; the question alone fills the model's input")
    if len(offsets) <= size:
        return [text]
    pieces = []
    for start in range(0, min(len(offsets), size * limit), size):
        chunk = offsets[start : start + size]
        pieces.append(text[chunk[0][0] : chunk[-1][1]])
    return pieces


def best_per_owner(scores: np.ndarray, owner: list[int], n: int) -> np.ndarray:
    """The highest score among each page's windows."""
    out = np.full(n, -np.inf)
    for s, o in zip(scores, owner, strict=True):
        out[o] = max(out[o], s)
    return out


def rerank(ranked: list[list], scores: np.ndarray) -> list[list]:
    """The first len(scores) pages by score, ties kept in their old order; the rest after."""
    head = ranked[: len(scores)]
    order = sorted(range(len(head)), key=lambda i: -scores[i])
    return [head[i] for i in order] + ranked[len(scores):]


def study(evaluation: dict, questions: list[Question], page_text: Callable[[str, int], str],
          score: Callable[[str, list[str]], np.ndarray], configs: str | list[str],
          windows: int = 1) -> dict:
    """Rerank each system's top 30 for every question; score each ranking before and after the
    same way. A page two systems both put in their top 30 for a question is scored once."""
    configs = [configs] if isinstance(configs, str) else list(configs)
    cache: dict[tuple, float] = {}
    pairs_scored, scoring_seconds = 0, 0.0
    per_q: dict[str, dict] = {}
    for q in questions:
        x = evaluation["questions"][q.qid]
        runs, tops = {}, {}
        for config in configs:
            base = [tuple(k) for k in x["runs"][config]["top"][:TOP]]
            missing = [k for k in dict.fromkeys(base) if (q.qid, *k) not in cache]
            if missing:
                t0 = time.perf_counter()
                new_scores = score(q.text, [page_text(d, p) for d, p in missing])
                scoring_seconds += time.perf_counter() - t0
                pairs_scored += len(missing)
                for k, v in zip(missing, new_scores, strict=True):
                    cache[(q.qid, *k)] = float(v)
            scores = np.array([cache[(q.qid, *k)] for k in base])
            new = rerank([list(k) for k in base], scores)
            runs[config] = score_ranking(base, q)
            runs[f"{config}+rerank"] = score_ranking([tuple(k) for k in new], q)
            tops[config] = new[:10]
        per_q[q.qid] = {"split": x["split"], "type": x["type"], "runs": runs, "top_after": tops}
    names = [n for c in configs for n in (c, f"{c}+rerank")]
    table = []
    for name in names:
        for part in ("dev", "test"):
            recs = [v["runs"][name] for v in per_q.values() if v["split"] == part]
            if not recs:
                continue
            row = {"config": name, "split": part, "n": len(recs), **metrics(recs)}
            for m in ("hit@10", "mrr@10"):
                row[f"{m}_ci95"] = list(bootstrap_ci(per_question(recs, m)))
            table.append(row)
    res = {"bases": configs, "base": configs[0], "top": TOP, "windows": windows, "table": table,
           "questions": per_q, "pairs_scored": pairs_scored,
           "scoring_seconds": round(scoring_seconds, 3),
           "seconds_per_pair": scoring_seconds / pairs_scored if pairs_scored else None}
    parts = [part for part in ("dev", "test") if any(v["split"] == part for v in per_q.values())]
    comps = [compare(res, f"{c}+rerank", c, part, m) for c in configs for part in parts
             for m in ("hit@1", "hit@10", "mrr@10")]
    comps += [compare(res, f"{c}+rerank", other, part, m) for c in configs for other in configs
              if other != c for part in parts for m in ("hit@10", "mrr@10")]
    res["comparisons"] = comps
    return res
