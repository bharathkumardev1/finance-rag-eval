"""Page-level retrieval evaluation on FinanceBench.

Every question is labeled with the filing and the page(s) that answer it, so a ranked list of
pages can be scored without reading generated answers:

* hit@k: at least one evidence page is in the top k
* recall@k: share of the question's evidence pages in the top k
* mrr@10: 1 / rank of the first evidence page, 0 if it isn't in the top 10
* doc_hit@k: some page of the right filing is in the top k

The 150 questions are split once into dev (50) and test (100), stratified by question type.
Anything I chose by looking at results (the alias table, any reranking) was chosen on dev.
Headline numbers are on test, with 95% bootstrap intervals.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import Question
from .search import RRF_DEPTH, Options, Searcher

KS = (1, 5, 10)
DEV_PER_TYPE = {"domain-relevant": 17, "metrics-generated": 17, "novel-generated": 16}
BOOTSTRAP = 10_000


def make_split(questions: list[Question], dev_per_type: dict[str, int] | None = None
               ) -> dict[str, list[str]]:
    """Deterministic stratified split: within each question type, order by sha256(id)."""
    dev_per_type = dev_per_type or DEV_PER_TYPE
    by_type: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        by_type[q.question_type].append(q)
    dev: list[str] = []
    for t in sorted(by_type):
        group = sorted(by_type[t], key=lambda q: hashlib.sha256(q.qid.encode()).hexdigest())
        dev += [q.qid for q in group[: dev_per_type[t]]]
    dev_set = set(dev)
    test = [q.qid for q in questions if q.qid not in dev_set]
    return {"dev": sorted(dev), "test": sorted(test)}


def gold_pages(q: Question) -> set[tuple[str, int]]:
    return {(e.doc_name, e.page) for e in q.evidence}


def score_ranking(rows: list[int], searcher: Searcher, q: Question) -> dict:
    """Ranks (1-based) of the evidence pages and of the first page from the right filing."""
    gold = gold_pages(q)
    docs = {d for d, _ in gold}
    c = searcher.corpus
    gold_ranks = [i + 1 for i, r in enumerate(rows) if (c.row_doc[r], c.row_page[r]) in gold]
    doc_rank = next((i + 1 for i, r in enumerate(rows) if c.row_doc[r] in docs), None)
    return {"gold_ranks": gold_ranks, "n_gold": len(gold), "doc_rank": doc_rank}


def metrics(records: list[dict], ks=KS) -> dict[str, float]:
    out: dict[str, float] = {}
    n = len(records)
    if n == 0:
        return out
    for k in ks:
        out[f"hit@{k}"] = sum(any(r <= k for r in x["gold_ranks"]) for x in records) / n
        out[f"recall@{k}"] = sum(
            sum(r <= k for r in x["gold_ranks"]) / x["n_gold"] for x in records) / n
        out[f"doc_hit@{k}"] = sum(
            x["doc_rank"] is not None and x["doc_rank"] <= k for x in records) / n
    out["mrr@10"] = sum(
        (1 / min(x["gold_ranks"])) if x["gold_ranks"] and min(x["gold_ranks"]) <= 10 else 0.0
        for x in records) / n
    return out


def per_question(records: list[dict], name: str) -> np.ndarray:
    """One number per question for ``name`` (e.g. hit@10), for bootstrapping."""
    metric, k = (name.split("@") + ["10"])[:2]
    k = int(k)
    vals = []
    for x in records:
        ranks = x["gold_ranks"]
        if metric == "hit":
            vals.append(float(any(r <= k for r in ranks)))
        elif metric == "recall":
            vals.append(sum(r <= k for r in ranks) / x["n_gold"])
        elif metric == "doc_hit":
            vals.append(float(x["doc_rank"] is not None and x["doc_rank"] <= k))
        elif metric == "mrr":
            vals.append(1 / min(ranks) if ranks and min(ranks) <= k else 0.0)
        else:
            raise ValueError(name)
    return np.asarray(vals)


def bootstrap_ci(values: np.ndarray, n: int = BOOTSTRAP, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_diff(a: np.ndarray, b: np.ndarray, n: int = BOOTSTRAP, seed: int = 0) -> dict:
    """Mean of a - b over the same questions, with a 95% paired bootstrap interval."""
    d = a - b
    lo, hi = bootstrap_ci(d, n, seed)
    return {"diff": float(d.mean()), "ci95": [lo, hi], "a_better": int((d > 0).sum()),
            "b_better": int((d < 0).sum())}


def filter_report(searcher: Searcher, questions: list[Question]) -> dict:
    f = searcher.filter
    found = kept = 0
    sizes = []
    for q in questions:
        if q.company in f.companies_in(q.text):
            found += 1
        docs = f.allowed_docs(q.text, "company_year")
        if docs is not None:
            sizes.append(len(docs))
            kept += q.doc_name in docs
    return {"n": len(questions), "company_found": found, "filtered": len(sizes),
            "gold_filing_kept": kept, "mean_filings_after": float(np.mean(sizes)) if sizes else None}


def latency_summary(seconds: list[float]) -> dict[str, float]:
    ms = np.asarray(seconds) * 1000
    return {"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
            "mean_ms": float(ms.mean())}


def ladder(methods: tuple[str, ...]) -> list[tuple[str, Options]]:
    """Each configuration adds one piece to the one before it, per retriever.

    Routing goes before dropping the company's name: the drop only helps once routing is
    in place (on its own it hurts), so the other order would credit routing with the
    damage the drop does without it.
    """
    steps = [
        Options("none", False, False, False),
        Options("company", False, False, False),
        Options("company_year", False, False, False),
        Options("company_year", True, False, False),
        Options("company_year", True, False, True),
        Options("company_year", True, True, True),
    ]
    out = []
    for m in methods:
        for o in steps:
            if m == "dense" and o.drop_company_terms:
                o = Options(o.filter, o.quarters, False, o.route_statements)
            if (m, o) not in out:
                out.append((m, o))
    return out


def config_key(method: str, opts: Options) -> str:
    return f"{method}/{opts.key()}"


def run(searcher: Searcher, questions: list[Question], split: dict[str, list[str]],
        configs: list[tuple[str, Options]] | None = None) -> dict:
    configs = configs or ladder(searcher.methods)
    q_by_id = {q.qid: q for q in questions}
    split_of = {qid: name for name, ids in split.items() for qid in ids}
    per_q: dict[str, dict] = {
        q.qid: {"split": split_of[q.qid], "type": q.question_type, "doc": q.doc_name,
                "n_gold": len(gold_pages(q)), "runs": {}}
        for q in questions
    }
    timings: dict[str, list[float]] = defaultdict(list)
    for q in questions:
        qvec, t_encode = None, 0.0
        if searcher.encoder is not None:
            t0 = time.perf_counter()
            qvec = searcher.encoder.encode([q.text])[0]
            t_encode = time.perf_counter() - t0
            timings["encode"].append(t_encode)
        for method, opts in configs:
            t0 = time.perf_counter()
            _, rows = searcher.filtered(q.text, opts.filter, opts.quarters)
            ranked, _ = searcher.rank(q.text, method, rows, RRF_DEPTH, qvec, opts)
            elapsed = time.perf_counter() - t0 + (t_encode if method != "bm25" else 0.0)
            key = config_key(method, opts)
            timings[key].append(elapsed)
            per_q[q.qid]["runs"][key] = score_ranking(ranked.tolist(), searcher, q)

    table = []
    for method, opts in configs:
        key = config_key(method, opts)
        for name in ("dev", "test", "all"):
            ids = [i for i in per_q if name == "all" or per_q[i]["split"] == name]
            recs = [per_q[i]["runs"][key] for i in ids]
            row = {"config": key, "method": method, **asdict(opts), "split": name, "n": len(recs),
                   **metrics(recs)}
            for m in ("hit@10", "mrr@10", "recall@10"):
                row[f"{m}_ci95"] = list(bootstrap_ci(per_question(recs, m)))
            row.update(latency_summary(timings[key]))
            table.append(row)
    return {
        "meta": run_meta(searcher, questions, split),
        "filter": {name: filter_report(searcher, [q_by_id[i] for i in ids])
                   for name, ids in split.items()},
        "table": table,
        "encode_latency": latency_summary(timings["encode"]) if timings.get("encode") else None,
        "questions": per_q,
    }


def compare(results: dict, a: str, b: str, split: str = "test", metric: str = "hit@10") -> dict:
    """Paired bootstrap of ``a`` minus ``b`` (keys like "hybrid/company_year")."""
    recs = [(x["runs"][a], x["runs"][b]) for x in results["questions"].values()
            if split == "all" or x["split"] == split]
    va = per_question([r[0] for r in recs], metric)
    vb = per_question([r[1] for r in recs], metric)
    return {"a": a, "b": b, "split": split, "metric": metric, **paired_diff(va, vb)}


def run_meta(searcher: Searcher, questions: list[Question], split: dict) -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    import onnxruntime
    import pymupdf
    import scipy

    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor() or None
    import os

    return {
        "git_sha": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "date": time.strftime("%Y-%m-%d"),
        "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__, "onnxruntime": onnxruntime.__version__,
        "pymupdf": pymupdf.__version__,
        "cpu": cpu, "cpus": os.cpu_count(),
        "filings": len(searcher.corpus.docs), "pages": len(searcher.corpus),
        "dense_windows": int(len(searcher.dense.window_page)) if searcher.dense is not None else None,
        "questions": len(questions), "dev": len(split["dev"]), "test": len(split["test"]),
        "rrf": {"k": 60, "depth": RRF_DEPTH}, "bootstrap": BOOTSTRAP,
    }


def save(results: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(results, indent=1) + "\n")
