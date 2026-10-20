"""Everything ``finrag eval`` runs, in one place, so the results file is reproducible."""

from __future__ import annotations

import json
from pathlib import Path

from . import evaluate as ev
from .data import load_questions
from .failures import taxonomy
from .search import Options, Searcher

FULL = Options()


def ablations(method: str) -> list[tuple[str, Options]]:
    """The full system with one piece taken out at a time."""
    out = [
        (method, Options(route_statements=False)),
        (method, Options(quarters=False)),
        (method, Options(filter="company")),
    ]
    if method != "dense":
        out.insert(0, (method, Options(drop_company_terms=False)))
    return out


def configs_for(methods: tuple[str, ...]) -> list[tuple[str, Options]]:
    seen: list[tuple[str, Options]] = []
    for item in ev.ladder(methods) + [a for m in methods for a in ablations(m)]:
        if item not in seen:
            seen.append(item)
    return seen


def full_key(method: str) -> str:
    opts = Options(drop_company_terms=method != "dense")
    return ev.config_key(method, opts)


def pick_headline(results: dict, methods: tuple[str, ...]) -> str:
    """Best full configuration on the dev split (hit@10, then MRR@10). Test plays no part."""
    rows = {r["config"]: r for r in results["table"] if r["split"] == "dev"}
    return max((full_key(m) for m in methods),
               key=lambda k: (rows[k]["hit@10"], rows[k]["mrr@10"]))


def run_all(searcher: Searcher, data_dir: Path, split_path: Path, out_dir: Path) -> Path:
    questions = load_questions(data_dir)
    split = json.loads(Path(split_path).read_text())
    methods = searcher.methods
    results = ev.run(searcher, questions, split, configs_for(methods))
    headline = pick_headline(results, methods)
    head_method = headline.split("/", 1)[0]
    head_opts = Options(drop_company_terms=head_method != "dense")

    comparisons = [
        (headline, "bm25/none"),
        ("bm25/company_year+quarter", "bm25/none"),
        (full_key("bm25"), ev.config_key("bm25", Options(route_statements=False))),
        (full_key("bm25"), ev.config_key("bm25", Options(drop_company_terms=False))),
    ]
    if "dense" in methods:
        comparisons += [
            (full_key("hybrid"), full_key("bm25")),
            (full_key("dense"), full_key("bm25")),
            ("dense/none", "bm25/none"),
            ("hybrid/none", "bm25/none"),
            (full_key("hybrid"), ev.config_key("hybrid", Options(route_statements=False))),
        ]
    results["comparisons"] = [
        ev.compare(results, a, b, "test", metric)
        for a, b in comparisons for metric in ("hit@10", "mrr@10")
    ]
    results["headline"] = headline
    results["taxonomy"] = {
        "test": taxonomy(searcher, questions, results, headline, head_opts, "test"),
        "all": taxonomy(searcher, questions, results, headline, head_opts, None),
    }
    by_type: dict[str, dict] = {}
    for t in sorted({x["type"] for x in results["questions"].values()}):
        recs = [x["runs"][headline] for x in results["questions"].values()
                if x["split"] == "test" and x["type"] == t]
        base = [x["runs"]["bm25/none"] for x in results["questions"].values()
                if x["split"] == "test" and x["type"] == t]
        by_type[t] = {"n": len(recs), "headline": ev.metrics(recs), "bm25/none": ev.metrics(base)}
    results["by_type"] = by_type

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "eval.json"
    ev.save(results, path)
    from .report import write_markdown

    write_markdown(results, out_dir / "RESULTS.md")
    return path
