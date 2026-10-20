"""Export what the in-browser demo (``web/``) needs to run the BM25 pipeline client-side.

The browser can't hold 53,399 pages, so the demo ships the 84 filings that FinanceBench
questions cite, one gzipped JSON per filing, loaded only when the filter picks it. Scores
must match the Python benchmark exactly, so each filing also carries the corpus-wide IDF of
every term on its pages, and meta.json carries the corpus page count and average length.

``parity.json`` holds Python's own answers for the benchmark questions (allowed filings,
top-10 pages restricted to demo filings, and their float32 scores) so ``web/test`` can check
the JavaScript port against them.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from . import filters, statements
from .bm25 import K1, STOPWORDS, B, tokenize
from .data import load_questions
from .evaluate import gold_pages
from .filters import _DOC_QUARTER
from .search import FULL, Options, Searcher
from .statements import KINDS

full_bm25 = "bm25/company_year+quarter+drop+route"


def export(searcher: Searcher, data_dir: Path, out: Path, docs: list[str] | None = None,
           results_path: Path | None = None) -> dict:
    out = Path(out)
    (out / "filings").mkdir(parents=True, exist_ok=True)
    c = searcher.corpus
    questions = load_questions(data_dir)
    docs = sorted(docs or {q.doc_name for q in questions})
    bm = searcher.bm25

    sizes = {}
    for doc in docs:
        start, end = c.offsets[doc]
        pages = c.texts[start:end]
        lengths = [int(bm.lengths[r]) for r in range(start, end)]
        kinds = ["".join(k[0] for k in KINDS if k in searcher.kinds[r]) for r in range(start, end)]
        vocab = sorted({t for page in pages for t in tokenize(page)})
        idf = {t: float(bm.idf[bm.vocab[t]]) for t in vocab}
        blob = json.dumps({"doc": doc, "pages": pages, "lengths": lengths, "kinds": kinds, "idf": idf},
                          ensure_ascii=False, separators=(",", ":")).encode("utf8")
        gz = gzip.compress(blob, compresslevel=9, mtime=0)
        (out / "filings" / f"{doc}.json.gz").write_bytes(gz)
        sizes[doc] = len(gz)

    metas = c.metas
    filings = []
    for d in c.docs:
        m = metas[d]
        q = _DOC_QUARTER.search(d + "_")
        filings.append({"doc": d, "company": m.company, "period": m.period, "type": m.doc_type,
                        "quarter": [int(q.group(1)), int(q.group(2))] if q else None,
                        "pages": c.offsets[d][1] - c.offsets[d][0], "demo": d in sizes,
                        "gz_bytes": sizes.get(d)})
    meta = {
        "n_pages": len(c), "avg_len": bm.avg_len, "k1": K1, "b": B,
        "stopwords": sorted(STOPWORDS), "aliases": [list(a) for a in searcher.filter.aliases],
        "filings": filings, "kinds": {k[0]: k for k in KINDS},
        # Regex sources shared with the browser, so both sides parse questions identically.
        "patterns": {
            "year": filters._YEAR.pattern, "short_fy": filters._SHORT_FY.pattern,
            "quarter": [rx.pattern for rx in filters._QUARTER],
            "ordinal": filters._ORDINAL,
            "named": statements._NAMED, "implied": statements._IMPLIED,
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))

    split = {}
    split_path = Path("configs/split.json")
    if split_path.exists():
        s = json.loads(split_path.read_text())
        split = {qid: name for name, ids in s.items() for qid in ids}
    ranks = {}
    res = None
    if results_path and Path(results_path).exists():
        res = json.loads(Path(results_path).read_text())
        keep = {"none": "bm25/none", "bm25": full_bm25, "headline": res.get("headline")}
        for qid, x in res["questions"].items():
            ranks[qid] = {name: (min(x["runs"][k]["gold_ranks"]) if x["runs"][k]["gold_ranks"] else None)
                          for name, k in keep.items() if k in x["runs"]}
    qs_out, parity = [], []
    demo_rows = np.concatenate([np.arange(*c.offsets[d]) for d in docs])
    for q in questions:
        qs_out.append({"id": q.qid, "q": q.text, "a": q.answer, "type": q.question_type,
                       "split": split.get(q.qid), "gold": sorted([d, p] for d, p in gold_pages(q)),
                       "ranks": ranks.get(q.qid, {})})
        for opts in (FULL, Options(route_statements=False), Options(drop_company_terms=False)):
            allowed, rows = searcher.filtered(q.text, opts.filter, opts.quarters)
            if rows is None:
                continue
            rows = np.intersect1d(rows, demo_rows)
            if not len(rows):
                continue
            ranked, scores = searcher.rank(q.text, "bm25", rows, 10, opts=opts)
            parity.append({"id": q.qid, "opts": {"route": opts.route_statements,
                                                  "drop": opts.drop_company_terms},
                           "allowed": sorted(allowed),
                           "top": [[c.row_doc[r], c.row_page[r]] for r in ranked.tolist()],
                           "scores": [float(np.float32(s)) for s in scores.tolist()]})
    (out / "questions.json").write_text(json.dumps(qs_out, ensure_ascii=False, separators=(",", ":")))
    (out / "parity.json").write_text(json.dumps(parity, separators=(",", ":")))
    if res is not None:
        (out / "summary.json").write_text(json.dumps(summary(res, parity, results_path), indent=1))
    return {"filings": len(docs), "gz_mb": round(sum(sizes.values()) / 2**20, 2),
            "parity_cases": len(parity)}


def summary(res: dict, parity: list[dict], results_path: Path) -> dict:
    """The numbers the portfolio page prints, taken straight from eval.json."""
    test = {r["config"]: r for r in res["table"] if r["split"] == "test"}
    keys = ("hit@1", "hit@5", "hit@10", "hit@10_ci95", "mrr@10", "recall@10", "doc_hit@10",
            "p50_ms", "p95_ms", "n")
    out = {
        "source": str(results_path), "git_sha": res["meta"]["git_sha"], "date": res["meta"]["date"],
        "meta": res["meta"], "headline": res["headline"],
        "test": {k: {m: v[m] for m in keys} for k, v in test.items()},
        "filter": res["filter"], "taxonomy": {k: v["counts"] | {"misses": v["misses"],
                                                            "questions": v["questions"]}
                                              for k, v in res["taxonomy"].items()},
        "comparisons": res["comparisons"],
        "parity": {"cases": len(parity), "questions": len({p["id"] for p in parity})},
    }
    lt = Path(results_path).with_name("loadtest.json")
    if lt.exists():
        out["loadtest"] = json.loads(lt.read_text())
    return out
