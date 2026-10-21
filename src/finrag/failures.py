"""Put every miss in the first stage of the pipeline that lost the gold page.

A miss is a question with no gold page in the top 10. The categories follow the pipeline,
and a miss goes in the first one that applies:

1. ``no_company``: the question names no company the filter knows, so search ran unfiltered.
2. ``filter_dropped_filing``: the filter kept other filings and dropped the one with the answer.
3. ``statement_not_detected``: the gold page is a financial statement, but nothing in the
   question pointed to that statement, so routing didn't promote it.
4. ``statement_outranked``: routing promoted the right kind of statement, but other pages of
   that kind (other filings, statements that run over several pages) came first.
5. ``other_filing_won``: the filter kept several filings and pages from the wrong ones took
   the top 10 (usually a question with no year in it).
6. ``vocabulary_gap``: a narrative page that shares at most two words with the question once
   the company's name is removed.
7. ``outranked``: a narrative page that shares more words than that with the question and
   still lost to pages that share more.

The categories are computed, not hand-labeled, so they come out of every evaluation run.
"""

from __future__ import annotations

from collections import Counter

from .bm25 import tokenize
from .data import Question
from .metrics import gold_pages
from .search import Options, Searcher
from .statements import question_kinds

CATEGORIES = (
    "no_company", "filter_dropped_filing", "statement_not_detected", "statement_outranked",
    "other_filing_won", "vocabulary_gap", "outranked",
)
MANY_FILINGS = 3
LOW_OVERLAP = 2


def classify(searcher: Searcher, q: Question, ranked: list[int], opts: Options) -> str:
    c = searcher.corpus
    docs, _ = searcher.filtered(q.text, opts)
    gold = sorted(gold_pages(q))
    gold_rows = [c.offsets[d][0] + p for d, p in gold if d in c.offsets]
    if docs is None and opts.filter != "none":
        return "no_company"
    if docs is not None and not any(d in docs for d, _ in gold):
        return "filter_dropped_filing"
    gold_kinds = set().union(*(searcher.kinds[r] for r in gold_rows)) if gold_rows else set()
    if gold_kinds:
        wanted = question_kinds(q.text) if opts.route_statements else frozenset()
        return "statement_outranked" if wanted & gold_kinds else "statement_not_detected"
    top_docs = [c.row_doc[r] for r in ranked[:10]]
    gold_docs = {d for d, _ in gold}
    if docs is not None and len(docs) >= MANY_FILINGS and sum(d in gold_docs for d in top_docs) <= 2:
        return "other_filing_won"
    words = set(tokenize(q.text)) - searcher.filter.company_terms(searcher.filter.companies_in(q.text))
    overlap = max(len(words & set(tokenize(c.texts[r]))) for r in gold_rows) if gold_rows else 0
    return "vocabulary_gap" if overlap <= LOW_OVERLAP else "outranked"


def taxonomy(searcher: Searcher, questions: list[Question], results: dict, config: str,
             opts: Options, split: str | None = "test") -> dict:
    """Misses of ``config`` per category, with the question ids in each."""
    by_id = {q.qid: q for q in questions}
    method = config.split("/", 1)[0]
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {k: [] for k in CATEGORIES}
    detail = []
    for qid, x in results["questions"].items():
        if split and x["split"] != split:
            continue
        run = x["runs"][config]
        if any(r <= 10 for r in run["gold_ranks"]):
            continue
        q = by_id[qid]
        ranked, _ = searcher.rank(q.text, 10, method, opts)
        cat = classify(searcher, q, ranked.tolist(), opts)
        counts[cat] += 1
        examples[cat].append(qid)
        detail.append({"qid": qid, "type": q.question_type, "category": cat,
                       "first_gold_rank": min(run["gold_ranks"]) if run["gold_ranks"] else None})
    n = sum(1 for x in results["questions"].values() if not split or x["split"] == split)
    return {"config": config, "split": split, "questions": n, "misses": sum(counts.values()),
            "counts": {k: counts.get(k, 0) for k in CATEGORIES}, "examples": examples,
            "misses_detail": detail}
