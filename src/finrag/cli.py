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


def cmd_model(args: argparse.Namespace) -> int:
    from .dense import download_model

    print(download_model(Path(args.model_dir)))
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    from .dense import MiniLM, embed_corpus
    from .evaluate import git

    model = MiniLM(Path(args.model_dir), threads=args.threads)
    print(json.dumps(embed_corpus(model, Path(args.data_dir), Path(args.emb_dir),
                                  git_sha=git("rev-parse", "HEAD"))))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from . import experiments as ex
    from .data import FINANCEBENCH_COMMIT, load_questions
    from .evaluate import DEPTH, run, run_meta, save
    from .failures import taxonomy
    from .filters import load_aliases
    from .report import write_markdown
    from .search import Searcher
    from .split import load_split

    data_dir = Path(args.data_dir)
    methods = tuple(args.methods.split(","))
    if methods[0] != "bm25":
        raise SystemExit("--methods must start with bm25; every comparison is against it")
    dense = len(methods) > 1
    aliases = load_aliases(Path(args.aliases)) if args.aliases else {}
    searcher = Searcher.build(data_dir, aliases, model_dir=Path(args.model_dir) if dense else None,
                              emb_dir=Path(args.emb_dir) if dense else None)
    questions = load_questions(data_dir)
    split = load_split(Path(args.split))
    for m in methods:  # the first call pays for page faults; keep it out of p95
        searcher.ranked_pages("warm up", 1, m)
    results = run(ex.systems(searcher, methods), questions, split)
    results["headline"] = headline = ex.pick_headline(results, methods)
    results["comparisons"] = ex.comparisons(results, methods, headline)
    results["filter"] = {name: ex.filter_report(searcher, [q for q in questions if q.qid in set(ids)])
                         for name, ids in split.items()}
    results["by_type"] = ex.by_type(results, headline)
    head_method = headline.split("/", 1)[0]
    results["taxonomy"] = {part: taxonomy(searcher, questions, results, headline,
                                          ex.full_for(head_method), part if part != "all" else None)
                           for part in ("test", "all")}
    extra = {}
    if dense:
        import onnxruntime

        extra = {"onnxruntime": onnxruntime.__version__, "dense_windows": len(searcher.dense),
                 "model_sha256": searcher.encoder.sha256,
                 "tokenizer_sha256": searcher.encoder.tokenizer_sha256}
    meta = run_meta(args.command_line, financebench_commit=FINANCEBENCH_COMMIT,
                    filings=len(searcher.corpus.docs), pages=len(searcher.corpus),
                    questions=len(questions), dev=len(split["dev"]), test=len(split["test"]),
                    depth=DEPTH, aliases=args.aliases, **extra)
    results = {"meta": meta, **results}
    save(results, Path(args.out))
    write_markdown(results, Path(args.markdown))
    for r in results["table"]:
        if r["split"] == "test":
            print(f"{r['config']:<48} hit@10 {r['hit@10']:.2f}  mrr@10 {r['mrr@10']:.3f}  "
                  f"p95 {r['p95_ms']:.1f} ms")
    print(f"headline (picked on dev): {headline}")
    return 0


def cmd_check_incremental(args: argparse.Namespace) -> int:
    from .data import load_questions
    from .evaluate import run_meta, save
    from .filters import MetadataFilter, load_aliases
    from .incremental import check
    from .search import Corpus

    data_dir = Path(args.data_dir)
    corpus = Corpus.load(data_dir)
    flt = MetadataFilter(corpus.metas, load_aliases(Path(args.aliases)))
    result = check(corpus, load_questions(data_dir), flt, args.holdout_every, args.batch)
    save({"meta": run_meta(args.command_line), **result}, Path(args.out))
    print(json.dumps(result))
    return 0 if result["passed"] else 1


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

    p = sub.add_parser("model", help="download all-MiniLM-L6-v2 (ONNX) and check its SHA-256")
    p.add_argument("--model-dir", default="models/all-MiniLM-L6-v2")
    p.set_defaults(func=cmd_model)

    p = sub.add_parser("embed", help="MiniLM vectors for every page window; resumes where it stopped")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model-dir", default="models/all-MiniLM-L6-v2")
    p.add_argument("--emb-dir", default="data/embeddings")
    p.add_argument("--threads", type=int, default=2)
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("eval", help="rank pages for every question and score the rankings")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--split", default="configs/split.json")
    p.add_argument("--methods", default="bm25,dense,hybrid",
                   help="comma-separated; dense and hybrid need `finrag embed` first")
    p.add_argument("--model-dir", default="models/all-MiniLM-L6-v2")
    p.add_argument("--emb-dir", default="data/embeddings")
    p.add_argument("--aliases", default="configs/aliases.yaml", help="YAML file of extra company names")
    p.add_argument("--out", default="results/eval.json")
    p.add_argument("--markdown", default="results/RESULTS.md")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("check-incremental",
                       help="grow a BM25 index in batches and compare it with a one-pass build")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--aliases", default="configs/aliases.yaml")
    p.add_argument("--holdout-every", type=int, default=10)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--out", default="results/incremental.json")
    p.set_defaults(func=cmd_check_incremental)

    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    args.command_line = shlex.join(["finrag", *argv])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
