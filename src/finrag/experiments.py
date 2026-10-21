"""Every configuration `finrag eval` runs and every comparison it reports, in one place."""

from __future__ import annotations

import numpy as np

from .data import Question
from .evaluate import DEPTH, System, compare
from .metrics import metrics
from .search import ALL_PAGES, FULL, Options, Searcher

# Each step adds one piece to the step before it. Routing goes in before dropping the
# company's name: the drop only helps once routing is there (on its own it hurts), so the
# other order would charge routing for the damage the drop does without it.
LADDER = (
    ALL_PAGES,
    Options("company"),
    Options("company_year"),
    Options("company_year", quarters=True),
    Options("company_year", quarters=True, route_statements=True),
    FULL,
)


NO_ROUTE = Options("company_year", quarters=True, drop_company_terms=True)


def full_for(method: str) -> Options:
    """The full system. Dropping the company's name changes BM25's query, so dense skips it."""
    return FULL if method != "dense" else Options("company_year", True, False, True)


def ablations(method: str) -> list[Options]:
    """The full system with one piece taken out at a time."""
    full = full_for(method)
    out = [
        Options(full.filter, full.quarters, full.drop_company_terms, False),
        Options(full.filter, False, full.drop_company_terms, full.route_statements),
        Options("company", False, full.drop_company_terms, full.route_statements),
    ]
    if method != "dense":
        out.insert(0, Options(full.filter, full.quarters, False, full.route_statements))
    return out


def key(method: str, opts: Options) -> str:
    return f"{method}/{opts.key()}"


def configs(methods: tuple[str, ...]) -> list[tuple[str, Options]]:
    out: list[tuple[str, Options]] = []
    for m in methods:
        steps = [o if m != "dense" else Options(o.filter, o.quarters, False, o.route_statements)
                 for o in LADDER]
        for o in [*steps, *ablations(m)]:
            if (m, o) not in out:
                out.append((m, o))
    return out


def systems(searcher: Searcher, methods: tuple[str, ...]) -> dict[str, System]:
    def system(m: str, o: Options) -> System:
        return lambda text: searcher.ranked_pages(text, DEPTH, m, o)

    return {key(m, o): system(m, o) for m, o in configs(methods)}


def pick_headline(results: dict, methods: tuple[str, ...]) -> str:
    """The best full system on dev, by Hit@10 and then MRR@10. Test plays no part."""
    dev = {r["config"]: r for r in results["table"] if r["split"] == "dev"}
    return max((key(m, full_for(m)) for m in methods),
               key=lambda k: (dev[k]["hit@10"], dev[k]["mrr@10"]))


def comparisons(results: dict, methods: tuple[str, ...], headline: str) -> list[dict]:
    base = key("bm25", ALL_PAGES)
    filtered = key("bm25", Options("company_year", quarters=True))
    routed = key("bm25", Options("company_year", quarters=True, route_statements=True))
    pairs = [
        (headline, base),
        (filtered, base),                                   # the filter
        (routed, filtered),                                 # + statement routing
        (key("bm25", FULL), routed),                        # + dropping the company's name
        (key("bm25", FULL), key("bm25", NO_ROUTE)),       # routing taken out of the full system
    ]
    if "dense" in methods:
        pairs += [
            (key("dense", ALL_PAGES), base),
            (key("hybrid", ALL_PAGES), base),
            (key("dense", full_for("dense")), key("bm25", FULL)),
            (key("hybrid", FULL), key("bm25", FULL)),
        ]
    seen, out = set(), []
    for a, b in pairs:
        if a != b and (a, b) not in seen:
            seen.add((a, b))
            out += [compare(results, a, b, "test", m) for m in ("hit@10", "mrr@10")]
    return out


def by_type(results: dict, headline: str) -> dict:
    base = key("bm25", ALL_PAGES)
    out = {}
    test = [x for x in results["questions"].values() if x["split"] == "test"]
    for t in sorted({x["type"] for x in test}):
        group = [x for x in test if x["type"] == t]
        out[t] = {"n": len(group), "headline": metrics([x["runs"][headline] for x in group]),
                  base: metrics([x["runs"][base] for x in group])}
    return out


def filter_report(searcher: Searcher, questions: list[Question]) -> dict:
    """How often the filter recognizes the company and keeps the filing with the answer."""
    f = searcher.filter
    found = kept = 0
    sizes = []
    for q in questions:
        found += q.company in f.companies_in(q.text)
        docs = f.allowed_docs(q.text, FULL.filter, FULL.quarters)
        if docs is not None:
            sizes.append(len(docs))
            kept += q.doc_name in docs
    return {"n": len(questions), "company_found": found, "filtered": len(sizes),
            "gold_filing_kept": kept, "mean_filings_after": float(np.mean(sizes)) if sizes else 0.0}
