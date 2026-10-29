"""Export what the browser demo (``web/``) needs to run the BM25 pipeline client-side.

The browser can't hold 53,399 pages, so the demo ships the filings the FinanceBench
questions are about, one gzipped JSON per filing, fetched only when the filter picks it. To
score pages exactly as Python does, each filing carries the corpus-wide idf of every term on
its pages and meta.json carries the corpus page count and average page length, so the
browser runs the same float64 arithmetic in the same order.

``parity.json`` holds Python's own answers for every question whose filter lands on demo
filings: the filings kept, the top 10 pages within the demo filings, and their scores.
``web/test`` checks the JavaScript against it. ``summary.json`` is what the portfolio site
prints, copied out of results/*.json so the page never retypes a number.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from . import filters, statements
from .bm25 import K1, STOPWORDS, B, tokenize
from .data import Question
from .filters import _DOC_QUARTER
from .metrics import gold_pages
from .search import FULL, Options, Searcher
from .statements import KINDS

# The full system, then without routing, then without dropping the company's name.
PARITY_OPTIONS = (FULL, Options("company_year", True, True, False),
                  Options("company_year", True, False, True))


def export(searcher: Searcher, questions: list[Question], split: dict[str, list[str]], out: Path,
           results_dir: Path | None = None) -> dict:
    out = Path(out)
    (out / "filings").mkdir(parents=True, exist_ok=True)
    c, bm = searcher.corpus, searcher.bm25
    lengths = bm.page_lengths()
    docs = sorted({q.doc_name for q in questions} & set(c.docs))
    sizes = {}
    for doc in docs:
        start, end = c.offsets[doc]
        pages = c.texts[start:end]
        terms = sorted({t for page in pages for t in tokenize(page)})
        blob = {"doc": doc, "pages": pages, "lengths": lengths[start:end].tolist(),
                "kinds": ["".join(k[0] for k in KINDS if k in searcher.kinds[r]) for r in range(start, end)],
                "idf": {t: float(bm.idf(bm.vocab[t])) for t in terms}}
        gz = gzip.compress(json.dumps(blob, ensure_ascii=False, separators=(",", ":")).encode("utf8"),
                           compresslevel=9, mtime=0)
        (out / "filings" / f"{doc}.json.gz").write_bytes(gz)
        sizes[doc] = len(gz)

    filings = []
    for d in c.docs:
        m = searcher.filter.metas[d]
        quarter = _DOC_QUARTER.search(d + "_")
        filings.append({"doc": d, "company": m.company, "periods": list(m.periods), "type": m.doc_type,
                        "quarter": [int(quarter.group(1)), int(quarter.group(2))] if quarter else None,
                        "pages": c.offsets[d][1] - c.offsets[d][0], "demo": d in sizes,
                        "gz_bytes": sizes.get(d)})
    meta = {
        "n_pages": bm.n_docs, "avg_len": bm.avg_len, "k1": K1, "b": B, "stopwords": sorted(STOPWORDS),
        "aliases": [list(a) for a in searcher.filter.aliases], "filings": filings,
        "kinds": {k[0]: k for k in KINDS},
        # Regex sources shared with the browser, so both sides read questions the same way.
        "patterns": {"year": filters._YEAR.pattern, "short_fy": filters._SHORT_FY.pattern,
                     "quarter": [rx.pattern for rx in filters._QUARTER], "ordinal": filters._ORDINAL,
                     "named": statements._NAMED, "implied": statements._IMPLIED},
    }
    (out / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))

    split_of = {qid: name for name, ids in split.items() for qid in ids}
    first_rank = _first_ranks(results_dir)
    demo = set(sizes)
    qs_out, parity = [], []
    for q in questions:
        qs_out.append({"id": q.qid, "q": q.text, "a": q.answer, "type": q.question_type,
                       "split": split_of.get(q.qid), "gold": sorted([d, p] for d, p in gold_pages(q)),
                       "first_gold_rank": first_rank.get(q.qid, {})})
        for opts in PARITY_OPTIONS:
            allowed = searcher.filter.allowed_docs(q.text, opts.filter, opts.quarters)
            if allowed is None or not allowed & demo:
                continue
            rows, scores = searcher.rank(q.text, 10, "bm25", opts, within=demo)
            parity.append({"id": q.qid,
                           "opts": {"route": opts.route_statements, "drop": opts.drop_company_terms},
                           "allowed": sorted(allowed), "top": [list(c.key(int(r))) for r in rows],
                           "scores": [float(s) for s in scores]})
    (out / "questions.json").write_text(json.dumps(qs_out, ensure_ascii=False, separators=(",", ":")))
    (out / "parity.json").write_text(json.dumps(parity, separators=(",", ":")))
    if results_dir is not None:
        (out / "summary.json").write_text(json.dumps(summary(Path(results_dir)), indent=1))
    return {"filings": len(docs), "gz_mb": round(sum(sizes.values()) / 2**20, 2), "parity_cases": len(parity)}


def _first_ranks(results_dir: Path | None) -> dict[str, dict]:
    path = Path(results_dir) / "eval.json" if results_dir else None
    if not path or not path.exists():
        return {}
    res = json.loads(path.read_text())
    keep = {"bm25": "bm25/all-pages", "full_bm25": "bm25/" + FULL.key(), "headline": res.get("headline")}
    return {qid: {name: (min(x["runs"][k]["gold_ranks"]) if x["runs"][k]["gold_ranks"] else None)
                  for name, k in keep.items() if k in x["runs"]}
            for qid, x in res["questions"].items()}


def summary(results_dir: Path) -> dict:
    """The numbers the portfolio page prints, each with the file and commit it came from."""
    out = {}
    for name in ("eval", "incremental", "chunks", "load", "rerank", "rerank-windows3", "generation"):
        path = results_dir / f"{name}.json"
        if not path.exists():
            continue
        res = json.loads(path.read_text())
        entry = {"file": f"results/{name}.json", "git_sha": res["meta"]["git_sha"],
                 "command": res["meta"]["command"], "date": res["meta"]["date"],
                 "git_dirty": res["meta"].get("git_dirty")}
        if res["meta"].get("model"):
            entry["model"] = res["meta"]["model"]
        if "table" in res:
            entry["table"] = [{k: v for k, v in r.items()} for r in res["table"]]
        for key in ("headline", "comparisons", "filter", "summary", "levels", "units", "passed",
                    "max_abs_score_diff", "same_top100", "queries", "held_out_filings",
                    "full_build_seconds", "add_batch_seconds", "judge_check", "windows", "top", "bases",
                    "pairs_scored", "seconds_per_pair", "generator", "judge", "retrieval",
                    "naive_retrieval", "split", "recorded_spend_usd", "cap_usd"):
            if key in res:
                entry[key] = res[key]
        if "taxonomy" in res:
            entry["taxonomy"] = {k: {"counts": v["counts"], "misses": v["misses"],
                                     "questions": v["questions"]} for k, v in res["taxonomy"].items()}
        out[name] = entry
    return out
