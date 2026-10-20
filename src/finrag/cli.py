"""finrag: page-level retrieval over SEC filings, evaluated on FinanceBench.

  finrag download        questions + 360 filing PDFs (about 670 MB) from the FinanceBench repo
  finrag pages           extract page text with PyMuPDF into data/pages
  finrag verify          check labeled evidence pages line up with the extracted text
  finrag model           download all-MiniLM-L6-v2 (ONNX) and check its SHA-256
  finrag embed           embed every page (incremental; about 3.5 h on 2 CPU cores)
  finrag eval            run every configuration and write results/eval.json + RESULTS.md
  finrag ask "..."       search from the terminal
  finrag serve           HTTP service on :8000
  finrag loadtest        load-test a running service
  finrag export-demo     write the data files for the in-browser demo
  finrag gate            fail if fixture metrics drop below configs/gate.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _searcher(args, dense: bool = True):
    from .search import Searcher, load_aliases

    model = Path(getattr(args, "model_dir", "") or "missing")
    emb = Path(getattr(args, "emb_dir", "") or "missing")
    use_dense = dense and not getattr(args, "no_dense", False) and model.exists() and emb.exists()
    aliases = load_aliases(Path(args.aliases)) if Path(args.aliases).exists() else {}
    return Searcher.build(Path(args.data_dir), aliases, model if use_dense else None,
                          emb if use_dense else None)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="finrag", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, dense=True):
        sp.add_argument("--data-dir", default="data")
        sp.add_argument("--aliases", default="configs/aliases.yaml")
        if dense:
            sp.add_argument("--model-dir", default="models/all-MiniLM-L6-v2")
            sp.add_argument("--emb-dir", default="data/emb/minilm")

    d = sub.add_parser("download")
    d.add_argument("--data-dir", default="data")
    d.add_argument("--workers", type=int, default=4)
    pg = sub.add_parser("pages")
    pg.add_argument("--data-dir", default="data")
    pg.add_argument("--workers", type=int, default=2)
    v = sub.add_parser("verify")
    v.add_argument("--data-dir", default="data")
    m = sub.add_parser("model")
    m.add_argument("--model-dir", default="models/all-MiniLM-L6-v2")
    e = sub.add_parser("embed")
    common(e)
    e.add_argument("--threads", type=int, default=os.cpu_count() or 1)

    ev = sub.add_parser("eval")
    common(ev)
    ev.add_argument("--split", default="configs/split.json")
    ev.add_argument("--out", default="results")
    ev.add_argument("--no-dense", action="store_true", help="BM25 only")

    a = sub.add_parser("ask")
    common(a)
    a.add_argument("question")
    a.add_argument("--method", default="hybrid")
    a.add_argument("-k", type=int, default=5)

    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--workers", type=int, default=1)

    lt = sub.add_parser("loadtest")
    lt.add_argument("--url", default="http://127.0.0.1:8000")
    lt.add_argument("--data-dir", default="data")
    lt.add_argument("--method", default="hybrid")
    lt.add_argument("--levels", default="1,2,4,8,16,32")
    lt.add_argument("--seconds", type=float, default=30.0)
    lt.add_argument("--out", default="results/loadtest.json")

    x = sub.add_parser("export-demo")
    common(x, dense=False)
    x.add_argument("--out", required=True)
    x.add_argument("--results", default="results/eval.json",
                   help="eval.json to take per-question ranks from, if it exists")

    g = sub.add_parser("gate")
    g.add_argument("--results", default="ci-results/eval.json")
    g.add_argument("--baseline", default="configs/gate.json")

    args = p.parse_args(argv)
    from . import data

    if args.cmd == "download":
        res = data.download(Path(args.data_dir), args.workers)
        print(f"downloaded {len(res['downloaded'])} filings, missing {len(res['missing'])}")
        for name in res["missing"]:
            print(f"  missing: {name}")
        return 1 if res["missing"] else 0
    if args.cmd == "pages":
        counts = data.build_page_cache(Path(args.data_dir), args.workers)
        failed = [k for k, n in counts.items() if n < 0]
        print(f"parsed {len(counts) - len(failed)} filings, "
              f"{sum(n for n in counts.values() if n > 0):,} pages, {len(failed)} failed")
        return 1 if failed else 0
    if args.cmd == "verify":
        counts = data.verify_evidence_pages(Path(args.data_dir))
        print(", ".join(f"{k}={n}" for k, n in counts.items()))
        return 1 if counts["off_by_one"] or counts["not_found"] else 0
    if args.cmd == "model":
        from .dense import download_model

        print(f"model ok: {download_model(Path(args.model_dir))}")
        return 0
    if args.cmd == "embed":
        from .dense import MiniLM, embed_corpus

        model = MiniLM(Path(args.model_dir), threads=args.threads)
        print(embed_corpus(model, Path(args.data_dir), Path(args.emb_dir)))
        return 0
    if args.cmd == "eval":
        from .experiments import run_all

        out = run_all(_searcher(args), Path(args.data_dir), Path(args.split), Path(args.out))
        print(f"wrote {out}")
        return 0
    if args.cmd == "ask":
        searcher = _searcher(args)
        method = args.method if args.method in searcher.methods else "bm25"
        hits, info = searcher.search(args.question, method, args.k)
        print(json.dumps(info))
        for h in hits:
            text = " ".join(searcher.corpus.texts[h.row].split())[:160]
            print(f"{h.score:8.4f}  {h.doc} p{h.page}  {text}")
        return 0
    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("finrag.service:app", host=args.host, port=args.port, workers=args.workers)
        return 0
    if args.cmd == "loadtest":
        from . import loadtest

        questions = [q.text for q in data.load_questions(Path(args.data_dir))]
        levels = tuple(int(c) for c in args.levels.split(","))
        rows = loadtest.run(args.url, questions, args.method, levels, args.seconds)
        loadtest.save(rows, {"url": args.url, "method": args.method, "seconds": args.seconds},
                      Path(args.out))
        return 0
    if args.cmd == "export-demo":
        from .demo import export

        print(export(_searcher(args, dense=False), Path(args.data_dir), Path(args.out),
                     results_path=Path(args.results)))
        return 0
    if args.cmd == "gate":
        from .gate import check

        failures = check(Path(args.results), Path(args.baseline))
        if failures:
            print("quality gate FAILED:")
            for f in failures:
                print(f"  {f}")
            return 1
        print("quality gate passed")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
