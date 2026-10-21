"""Command line entry point: ``finrag <command>``."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


def cmd_download(args: argparse.Namespace) -> int:
    from .data import download

    only = set(json.loads(Path(args.filings).read_text())["filings"]) if args.filings else None
    summary = download(Path(args.data_dir), Path(args.manifest), only=only)
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
    write_markdown(results, Path(args.markdown), Path(args.out).parent)
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


def cmd_gate(args: argparse.Namespace) -> int:
    from . import gate as g

    path = Path(args.config)
    if args.define:
        from .data import load_questions
        from .split import load_split

        subset = g.define(load_questions(Path(args.data_dir)), load_split(Path(args.split))["dev"])
        g.save({"max_drop": args.max_drop, **subset}, path)
        print(f"{len(subset['questions'])} questions over {len(subset['filings'])} filings -> {path}")
        return 0
    gate = g.load(path)
    hits = g.run(Path(args.data_dir), gate, Path(args.aliases))
    if args.record:
        g.save(g.record(gate, hits), path)
        print(f"recorded Hit@10 {sum(hits.values())}/{len(hits)} -> {path}")
        return 0
    ok, report = g.check(gate, hits)
    print(report)
    return 0 if ok else 1


def cmd_chunk_study(args: argparse.Namespace) -> int:
    from .chunks import study
    from .data import load_questions
    from .evaluate import run_meta, save
    from .filters import load_aliases
    from .search import Searcher
    from .split import load_split

    data_dir = Path(args.data_dir)
    searcher = Searcher.build(data_dir, load_aliases(Path(args.aliases)))
    results = study(searcher, load_questions(data_dir), load_split(Path(args.split)))
    save({"meta": run_meta(args.command_line), **results}, Path(args.out))
    for r in results["table"]:
        if r["split"] == "test":
            print(f"{r['config']:<52} hit@10 {r['hit@10']:.2f}  mrr@10 {r['mrr@10']:.3f}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from .data import load_pages, load_questions
    from .evaluate import run_meta, save
    from .generate import Budget, Recorder, openai_client, run
    from .split import load_split

    data_dir = Path(args.data_dir)
    evaluation = json.loads(Path(args.eval).read_text())
    config = args.config or evaluation["headline"]
    wanted = set(load_split(Path(args.split))[args.on])
    questions = [q for q in load_questions(data_dir) if q.qid in wanted]
    retrieved = {qid: x["runs"][config]["top"] for qid, x in evaluation["questions"].items()}
    pages = load_pages(data_dir)
    recorder = Recorder(Path(args.record), None if args.replay else openai_client(), Budget(args.cap))
    res = run(questions, retrieved, lambda d, p: pages[d][p], recorder, args.generator, args.judge)
    res = {"meta": run_meta(args.command_line), "retrieval": config, "split": args.on, **res}
    old = Path(args.out)
    if old.exists() and "judge_check" in (prev := json.loads(old.read_text())):
        res["judge_check"] = prev["judge_check"]
    save(res, old)
    for setting, v in res["summary"]["settings"].items():
        print(f"{setting:<10} accuracy {v['accuracy']:.2f} ({v['n']} questions, "
              f"{v['declined']} declined)  ${v['cost_usd']:.4f}")
    print(f"calls made {res['calls']['made']}, replayed {res['calls']['replayed']}; "
          f"recorded spend ${res['recorded_spend_usd']:.4f} of ${res['cap_usd']:.2f}")
    if res["stopped"]:
        print(f"stopped early: {res['stopped']}", file=sys.stderr)
        return 2
    return 0


def cmd_label_judge(args: argparse.Namespace) -> int:
    import csv

    from .data import load_questions
    from .generate import label_sheet

    results = json.loads(Path(args.results).read_text())
    questions = {q.qid: q for q in load_questions(Path(args.data_dir))}
    rows = label_sheet(results, questions, args.n)
    with open(args.out, "w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} answers to label in {args.out}: fill your_verdict with correct or incorrect")
    return 0


def cmd_judge_agreement(args: argparse.Namespace) -> int:
    import csv

    from .evaluate import save
    from .generate import agreement

    path = Path(args.results)
    results = json.loads(path.read_text())
    with open(args.labels, newline="", encoding="utf8") as fh:
        check = agreement(results, list(csv.DictReader(fh)))
    results["judge_check"] = {"labels": args.labels, **check}
    save(results, path)
    print(json.dumps(check))
    return 0


def cmd_reranker(args: argparse.Namespace) -> int:
    from .rerank import download_reranker

    print(download_reranker(Path(args.model_dir)))
    return 0


def cmd_rerank_study(args: argparse.Namespace) -> int:
    from .data import load_pages, load_questions
    from .evaluate import run_meta, save
    from .rerank import MODEL_ID, ONNX_SHA256, REVISION, CrossEncoder, study

    data_dir = Path(args.data_dir)
    evaluation = json.loads(Path(args.eval).read_text())
    config = args.config or evaluation["headline"]
    questions = [q for q in load_questions(data_dir) if q.qid in evaluation["questions"]]
    pages = load_pages(data_dir)
    encoder = CrossEncoder(Path(args.model_dir), threads=args.threads)
    res = study(evaluation, questions, lambda d, p: pages[d][p], encoder.score, config)
    meta = run_meta(args.command_line, model=MODEL_ID, revision=REVISION, onnx_sha256=ONNX_SHA256,
                    tokenizer_sha256=encoder.tokenizer_sha256, threads=args.threads)
    save({"meta": meta, **res}, Path(args.out))
    for r in res["table"]:
        print(f"{r['split']:<5} {r['config']:<56} hit@10 {r['hit@10']:.2f}  mrr@10 {r['mrr@10']:.3f}")
    return 0


def cmd_loadtest(args: argparse.Namespace) -> int:
    from .data import load_questions
    from .evaluate import run_meta, save
    from .loadtest import USD_PER_VCPU_HOUR, run

    questions = [q.text for q in load_questions(Path(args.data_dir))]
    levels = tuple(int(c) for c in args.levels.split(","))
    rows = []
    for method in args.methods.split(","):
        rows += run(args.base, questions, method, levels, args.seconds)
    meta = run_meta(args.command_line, server=args.server_note, usd_per_vcpu_hour=USD_PER_VCPU_HOUR)
    save({"meta": meta, "levels": rows}, Path(args.out))
    return 0 if all(r["errors"] == 0 for r in rows) else 1


def cmd_export_demo(args: argparse.Namespace) -> int:
    from .data import load_questions
    from .demo import export
    from .filters import load_aliases
    from .search import Searcher
    from .split import load_split

    data_dir = Path(args.data_dir)
    aliases = load_aliases(Path(args.aliases)) if args.aliases else {}
    searcher = Searcher.build(data_dir, aliases)
    info = export(searcher, load_questions(data_dir), load_split(Path(args.split)), Path(args.out),
                  Path(args.results) if args.results else None)
    print(json.dumps(info))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import write_markdown

    results_dir = Path(args.results)
    results = json.loads((results_dir / "eval.json").read_text())
    write_markdown(results, results_dir / "RESULTS.md", results_dir)
    print(results_dir / "RESULTS.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finrag")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="FinanceBench questions, metadata and filing PDFs")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--manifest", default="configs/financebench.sha256")
    p.add_argument("--filings", default=None,
                   help="JSON file with a \"filings\" list; download only those PDFs")
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

    p = sub.add_parser("gate", help="regression gate: BM25 pipeline on the dev questions' filings")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--config", default="configs/gate.json")
    p.add_argument("--aliases", default="configs/aliases.yaml")
    p.add_argument("--split", default="configs/split.json")
    p.add_argument("--max-drop", type=int, default=1)
    p.add_argument("--define", action="store_true", help="write the question and filing lists")
    p.add_argument("--record", action="store_true", help="store this run as the one to compare with")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("chunk-study", help="BM25 over 128-, 256- and 650-token chunks against pages")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--aliases", default="configs/aliases.yaml")
    p.add_argument("--split", default="configs/split.json")
    p.add_argument("--out", default="results/chunks.json")
    p.set_defaults(func=cmd_chunk_study)

    p = sub.add_parser("generate", help="answer test questions from retrieved and gold pages, then judge")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--eval", default="results/eval.json", help="retrieval results to take pages from")
    p.add_argument("--config", default=None, help="retrieval system (default: the headline)")
    p.add_argument("--split", default="configs/split.json")
    p.add_argument("--on", default="test", choices=("dev", "test"))
    p.add_argument("--generator", default="gpt-6-luna")
    p.add_argument("--judge", default="gpt-6.1-sol")
    p.add_argument("--cap", type=float, default=25.0, help="USD, over everything in the recording")
    p.add_argument("--record", default="results/generation/calls.jsonl")
    p.add_argument("--replay", action="store_true", help="use recorded replies only; no API key needed")
    p.add_argument("--out", default="results/generation.json")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("label-judge", help="sample judged answers for a person to label")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results", default="results/generation.json")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--out", default="results/generation/labels.csv")
    p.set_defaults(func=cmd_label_judge)

    p = sub.add_parser("judge-agreement", help="how often the judge agrees with your labels")
    p.add_argument("--results", default="results/generation.json")
    p.add_argument("--labels", default="results/generation/labels.csv")
    p.set_defaults(func=cmd_judge_agreement)

    p = sub.add_parser("reranker", help="download BAAI/bge-reranker-base (ONNX, 1.1 GB), check its SHA-256")
    p.add_argument("--model-dir", default="models/bge-reranker-base")
    p.set_defaults(func=cmd_reranker)

    p = sub.add_parser("rerank-study", help="rerank a system's top 30 with the cross-encoder")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--eval", default="results/eval.json")
    p.add_argument("--config", default=None, help="retrieval system (default: the headline)")
    p.add_argument("--model-dir", default="models/bge-reranker-base")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="results/rerank.json")
    p.set_defaults(func=cmd_rerank_study)

    p = sub.add_parser("loadtest", help="closed-loop load test against a running service")
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--methods", default="bm25,hybrid")
    p.add_argument("--levels", default="1,2,4,8,16")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--server-note", default="", help="how the server ran, saved with the results")
    p.add_argument("--out", default="results/load.json")
    p.set_defaults(func=cmd_loadtest)

    p = sub.add_parser("export-demo", help="filings, metadata and parity cases for the browser demo")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--aliases", default="configs/aliases.yaml")
    p.add_argument("--split", default="configs/split.json")
    p.add_argument("--results", default="results", help="directory of results/*.json for summary.json")
    p.add_argument("--out", default="build/demo")
    p.set_defaults(func=cmd_export_demo)

    p = sub.add_parser("report", help="rewrite results/RESULTS.md from every results/*.json")
    p.add_argument("--results", default="results")
    p.set_defaults(func=cmd_report)

    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    args.command_line = shlex.join(["finrag", *argv])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
