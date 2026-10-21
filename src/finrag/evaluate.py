"""Run retrieval systems over the questions and record what they ranked and how fast.

A system is a function from question text to ranked (filing, page) keys. It never sees the
question's labels or metadata, only the text a user would type, so a filter or a reranker
can't lean on the answer key by accident.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .data import Question
from .metrics import BOOTSTRAP, PageKey, bootstrap_ci, metrics, paired_diff, per_question, score_ranking

DEPTH = 100  # pages kept per question; every metric looks at the top 10 or fewer
System = Callable[[str], list[PageKey]]


def latency_summary(seconds: list[float]) -> dict[str, float]:
    ms = np.asarray(seconds) * 1000
    return {"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
            "mean_ms": float(ms.mean())}


def run(systems: dict[str, System], questions: list[Question], split: dict[str, list[str]]) -> dict:
    """Rank every question with every system, one query at a time, and score the rankings."""
    split_of = {qid: name for name, ids in split.items() for qid in ids}
    per_q = {q.qid: {"split": split_of[q.qid], "type": q.question_type, "doc": q.doc_name,
                     "n_gold": len(q.evidence), "runs": {}} for q in questions}
    seconds: dict[str, dict[str, float]] = {name: {} for name in systems}
    for q in questions:
        for name, system in systems.items():
            t0 = time.perf_counter()
            ranked = system(q.text)[:DEPTH]
            seconds[name][q.qid] = time.perf_counter() - t0
            per_q[q.qid]["runs"][name] = {**score_ranking(ranked, q), "top10": [list(k) for k in ranked[:10]]}

    table = []
    for name in systems:
        for part in ("dev", "test", "all"):
            ids = [i for i, x in per_q.items() if part == "all" or x["split"] == part]
            recs = [per_q[i]["runs"][name] for i in ids]
            row = {"config": name, "split": part, "n": len(recs), **metrics(recs)}
            for m in ("hit@10", "mrr@10", "recall@10"):
                row[f"{m}_ci95"] = list(bootstrap_ci(per_question(recs, m)))
            row.update(latency_summary([seconds[name][i] for i in ids]))
            table.append(row)
    return {"table": table, "questions": per_q}


def compare(results: dict, a: str, b: str, split: str = "test", metric: str = "hit@10") -> dict:
    """Paired bootstrap of system ``a`` minus system ``b`` on the same questions."""
    pairs = [(x["runs"][a], x["runs"][b]) for x in results["questions"].values()
             if split == "all" or x["split"] == split]
    va = per_question([p[0] for p in pairs], metric)
    vb = per_question([p[1] for p in pairs], metric)
    return {"a": a, "b": b, "split": split, "metric": metric, **paired_diff(va, vb)}


def git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cpu_name() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def run_meta(command: str, **extra) -> dict:
    """Where a result came from: the commit, the command, the machine and the libraries."""
    import pymupdf
    import scipy

    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "command": command,
        "date": time.strftime("%Y-%m-%d"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pymupdf": pymupdf.__version__,
        "cpu": cpu_name(),
        "cpus": os.cpu_count(),
        "bootstrap": BOOTSTRAP,
        **extra,
    }


def save(results: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(results, indent=1) + "\n")
