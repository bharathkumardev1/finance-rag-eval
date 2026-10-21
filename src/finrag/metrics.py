"""Page-level retrieval metrics, bootstrap intervals and paired comparisons.

Every question is labeled with the page or pages that answer it, so a ranked list of pages
can be scored without reading any generated answer:

* hit@k        at least one gold page is in the top k
* recall@k     share of the question's gold pages in the top k (only differs from hit@k on
               questions with more than one gold page, so it's also reported on those alone)
* mrr@10       1 / rank of the first gold page, 0 when none is in the top 10
* doc_hit@k    some page of the right filing is in the top k
"""

from __future__ import annotations

import numpy as np

from .data import Question

KS = (1, 5, 10)
BOOTSTRAP = 10_000

PageKey = tuple[str, int]  # (filing name, zero-indexed page)


def gold_pages(q: Question) -> set[PageKey]:
    return {(e.doc_name, e.page) for e in q.evidence}


def score_ranking(ranked: list[PageKey], q: Question) -> dict:
    """1-based ranks of the gold pages in ``ranked``, and of the first page from a gold filing."""
    gold = gold_pages(q)
    docs = {d for d, _ in gold}
    gold_ranks = [i + 1 for i, key in enumerate(ranked) if key in gold]
    doc_rank = next((i + 1 for i, (d, _) in enumerate(ranked) if d in docs), None)
    return {"gold_ranks": gold_ranks, "n_gold": len(gold), "doc_rank": doc_rank}


def per_question(records: list[dict], name: str) -> np.ndarray:
    """One value per question for a metric like ``hit@10``, for averaging and bootstrapping."""
    metric, k = name.split("@")
    k = int(k)
    out = []
    for x in records:
        ranks = x["gold_ranks"]
        if metric == "hit":
            out.append(float(any(r <= k for r in ranks)))
        elif metric == "recall":
            out.append(sum(r <= k for r in ranks) / x["n_gold"])
        elif metric == "mrr":
            out.append(1 / min(ranks) if ranks and min(ranks) <= k else 0.0)
        elif metric == "doc_hit":
            out.append(float(x["doc_rank"] is not None and x["doc_rank"] <= k))
        else:
            raise ValueError(f"unknown metric {name!r}")
    return np.asarray(out, dtype=np.float64)


def metrics(records: list[dict], ks=KS) -> dict[str, float | int]:
    if not records:
        return {}
    out: dict[str, float | int] = {}
    for k in ks:
        for m in ("hit", "recall", "doc_hit"):
            out[f"{m}@{k}"] = float(per_question(records, f"{m}@{k}").mean())
    out["mrr@10"] = float(per_question(records, "mrr@10").mean())
    multi = [x for x in records if x["n_gold"] > 1]
    out["n_multi_page"] = len(multi)
    if multi:
        out["recall@10_multi_page"] = float(per_question(multi, "recall@10").mean())
    return out


def bootstrap_ci(values: np.ndarray, n: int = BOOTSTRAP, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of the mean, resampling questions with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    lo, hi = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def paired_diff(a: np.ndarray, b: np.ndarray, n: int = BOOTSTRAP, seed: int = 0) -> dict:
    """Mean of a - b over the same questions, with a paired bootstrap interval.

    Resampling the per-question differences keeps the pairing, which is what makes a small
    difference between two systems on the same 100 questions measurable at all.
    """
    d = a - b
    lo, hi = bootstrap_ci(d, n, seed)
    return {"diff": float(d.mean()), "ci95": [lo, hi],
            "a_better": int((d > 0).sum()), "b_better": int((d < 0).sum())}
