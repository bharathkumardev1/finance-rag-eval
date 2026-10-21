"""Command line entry point: ``finrag <command>``."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


def cmd_download(args: argparse.Namespace) -> int:
    from .data import download

    summary = download(Path(args.data_dir), Path(args.manifest))
    print(json.dumps({status: len(names) for status, names in summary.items()}))
    bad = summary.get("failed", []) + summary.get("mismatch", [])
    for name in bad:
        print(f"not downloaded: {name}", file=sys.stderr)
    return 1 if bad else 0


def cmd_pages(args: argparse.Namespace) -> int:
    from .data import build_page_cache

    counts = build_page_cache(Path(args.data_dir), workers=args.workers)
    failed = sorted(doc for doc, n in counts.items() if n < 0)
    print(json.dumps({"filings": len(counts), "pages": sum(n for n in counts.values() if n > 0),
                      "failed": failed}))
    return 1 if failed else 0


def cmd_verify(args: argparse.Namespace) -> int:
    from .data import verify_evidence_pages

    counts = verify_evidence_pages(Path(args.data_dir))
    print(json.dumps(counts))
    return 0 if counts["missing_doc"] == 0 else 1


def cmd_split(args: argparse.Namespace) -> int:
    from .data import load_questions
    from .split import DEV_PER_TYPE, make_split, save_split

    questions = load_questions(Path(args.data_dir))
    per_type = {t: args.per_type for t in DEV_PER_TYPE} if args.per_type else None
    split = make_split(questions, per_type)
    save_split(split, Path(args.out))
    print(json.dumps({name: len(ids) for name, ids in split.items()}))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .data import FINANCEBENCH_COMMIT, load_questions
    from .evaluate import DEPTH, run, run_meta, save
    from .report import write_markdown
    from .search import Searcher
    from .split import load_split

    data_dir = Path(args.data_dir)
    searcher = Searcher.build(data_dir)
    questions = load_questions(data_dir)
    split = load_split(Path(args.split))
    systems = {"bm25/all-pages": lambda text: searcher.ranked_pages(text, DEPTH)}
    searcher.ranked_pages("warm up", 1)  # first call pays for page faults; keep it out of p95
    results = run(systems, questions, split)
    meta = run_meta(args.command_line, financebench_commit=FINANCEBENCH_COMMIT,
                    filings=len(searcher.corpus.docs), pages=len(searcher.corpus),
                    questions=len(questions), dev=len(split["dev"]), test=len(split["test"]),
                    depth=DEPTH)
    results = {"meta": meta, **results}
    save(results, Path(args.out))
    write_markdown(results, Path(args.markdown))
    for r in results["table"]:
        if r["split"] == "test":
            print(f"{r['config']:<24} hit@10 {r['hit@10']:.2f}  mrr@10 {r['mrr@10']:.3f}  "
                  f"p95 {r['p95_ms']:.1f} ms")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finrag")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="FinanceBench questions, metadata and filing PDFs")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--manifest", default="configs/financebench.sha256")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("pages", help="extract page text from every PDF into data/pages")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--workers", type=int, default=2)
    p.set_defaults(func=cmd_pages)

    p = sub.add_parser("verify", help="check labeled evidence pages against the extracted text")
    p.add_argument("--data-dir", default="data")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("split", help="write the stratified dev/test split")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="configs/split.json")
    p.add_argument("--per-type", type=int, default=None,
                   help="dev questions per type (default: 17, 17 and 16 of the 150)")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("eval", help="rank pages for every question and score the rankings")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--split", default="configs/split.json")
    p.add_argument("--out", default="results/eval.json")
    p.add_argument("--markdown", default="results/RESULTS.md")
    p.set_defaults(func=cmd_eval)

    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    args.command_line = shlex.join(["finrag", *argv])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
