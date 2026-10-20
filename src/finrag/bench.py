"""Experiment runner: retrieval grid and approximate-nearest-neighbor comparison."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .chunking import Chunk, chunk_corpus
from .data import load_doc_meta, load_pages, load_questions
from .embedders import make_embedder
from .filters import MetadataFilter
from .metrics import aggregate, latency_summary, score_question
from .retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    build_faiss_index,
    set_search_param,
)


@dataclass
class BenchConfig:
    data_dir: str = "data"
    embedder: str = "lsa-256"
    chunkings: list[list[int]] = field(default_factory=lambda: [[512, 0]])
    methods: list[str] = field(default_factory=lambda: ["bm25", "dense", "hybrid"])
    filters: list[str] = field(default_factory=lambda: ["none", "company", "company_year"])
    ks: list[int] = field(default_factory=lambda: [1, 5, 10])
    hybrid_depth: int = 100
    min_doc_chars: int = 500
    company_aliases: dict[str, list[str]] = field(default_factory=dict)
    ann: dict | None = None


def load_config(path: str | Path) -> BenchConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf8")) or {}
    return BenchConfig(**raw)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _single_thread() -> None:
    import faiss

    faiss.omp_set_num_threads(1)


class Workspace:
    """Corpus, questions, and filter loaded once and shared by every experiment."""

    def __init__(self, cfg: BenchConfig):
        data_dir = Path(cfg.data_dir)
        metas = load_doc_meta(data_dir)
        pages = load_pages(data_dir)
        self.docs = {
            d: p for d, p in pages.items()
            if d in metas and sum(len(t) for t in p) >= cfg.min_doc_chars
        }
        self.metas = {d: metas[d] for d in self.docs}
        all_q = load_questions(data_dir)
        self.questions = [q for q in all_q if all(e.doc_name in self.docs for e in q.evidence)]
        self.skipped = [q.qid for q in all_q if q not in self.questions]
        self.filter = MetadataFilter(self.metas, cfg.company_aliases)

    def masks(self, chunks: list[Chunk], level: str) -> list[np.ndarray | None]:
        by_doc: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(chunks):
            by_doc[c.doc_name].append(i)
        out: list[np.ndarray | None] = []
        for q in self.questions:
            allowed = self.filter.allowed_docs(q.text, level)
            if allowed is None:
                out.append(None)
                continue
            m = np.zeros(len(chunks), dtype=bool)
            for d in allowed:
                m[by_doc.get(d, [])] = True
            out.append(m)
        return out

    def filter_report(self, level: str) -> dict:
        kept, matched, sizes = 0, 0, []
        for q in self.questions:
            allowed = self.filter.allowed_docs(q.text, level)
            gold = {e.doc_name for e in q.evidence}
            if allowed is None:
                sizes.append(len(self.docs))
                kept += 1
                continue
            matched += 1
            sizes.append(len(allowed))
            kept += gold <= allowed
        n = len(self.questions)
        return {
            "level": level,
            "questions_with_company_match": matched / n,
            "gold_filing_retained": kept / n,
            "mean_candidate_filings": float(np.mean(sizes)),
        }


def _run_method(retriever, ws: Workspace, chunks, masks, ks) -> tuple[dict, dict]:
    kmax = max(ks)
    per_q, lat, hits = [], [], {}
    for q, m in zip(ws.questions, masks, strict=True):
        t = time.perf_counter()
        ranking = retriever.search(q.text, kmax, m)
        lat.append(time.perf_counter() - t)
        s = score_question(ranking, chunks, q, tuple(ks))
        per_q.append(s)
        hits[q.qid] = s[f"hit@{kmax}"]
    return {**aggregate(per_q), **latency_summary(lat)}, hits


def run_benchmark(cfg: BenchConfig, out_dir: str | Path = "results") -> dict:
    _single_thread()
    ws = Workspace(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "meta": _meta(cfg, ws),
        "filters": [ws.filter_report(level) for level in cfg.filters],
        "rows": [],
        "per_question_hits": {},
        "question_types": {q.qid: q.question_type for q in ws.questions},
    }
    for size, overlap in cfg.chunkings:
        t0 = time.perf_counter()
        chunks = chunk_corpus(ws.docs, size, overlap)
        texts = [c.text for c in chunks]
        bm25 = BM25Retriever(texts)
        embedder = make_embedder(cfg.embedder)
        embedder.fit(texts)
        dense = DenseRetriever(embedder, embedder.encode(texts))
        retrievers = {
            "bm25": bm25,
            "dense": dense,
            "hybrid": HybridRetriever(bm25, dense, depth=cfg.hybrid_depth),
        }
        build_s = time.perf_counter() - t0
        for level in cfg.filters:
            masks = ws.masks(chunks, level)
            for method in cfg.methods:
                metrics, hits = _run_method(retrievers[method], ws, chunks, masks, cfg.ks)
                key = f"{size}/{overlap}/{method}/{level}"
                report["per_question_hits"][key] = hits
                report["rows"].append({
                    "chunk_size": size,
                    "overlap": overlap,
                    "method": method,
                    "embedder": "-" if method == "bm25" else embedder.name,
                    "filter": level,
                    "n_chunks": len(chunks),
                    "build_s": round(build_s, 1),
                    **{k: round(v, 4) for k, v in metrics.items()},
                })
                r = report["rows"][-1]
                print(f"{key:32s} hit@10={r['hit@10']:.3f} mrr@10={r['mrr@10']:.3f} "
                      f"p50={r['p50_ms']:.1f}ms", flush=True)
    (out_dir / "benchmark.json").write_text(json.dumps(report, indent=1), encoding="utf8")
    return report


def run_ann(cfg: BenchConfig, out_dir: str | Path = "results", repeats: int = 5) -> dict:
    """Exact vs approximate search on the same vectors: recall against exact, end-task hits,
    latency, build time, and index size."""
    import faiss

    _single_thread()
    ann = cfg.ann or {}
    size, overlap = ann.get("chunking", [512, 0])
    k = int(ann.get("k", 10))
    ws = Workspace(cfg)
    chunks = chunk_corpus(ws.docs, size, overlap)
    texts = [c.text for c in chunks]
    embedder = make_embedder(cfg.embedder)
    embedder.fit(texts)
    vecs = embedder.encode(texts)
    qvecs = embedder.encode([q.text for q in ws.questions])

    exact = build_faiss_index(vecs, "flat")
    _, exact_ids = exact.search(qvecs, k)

    variants: list[tuple[str, dict, str | None, list]] = [("flat", {}, None, [None])]
    variants.append(("hnsw", {"M": ann.get("hnsw_M", 32),
                              "ef_construction": ann.get("hnsw_ef_construction", 200)},
                     "efSearch", ann.get("hnsw_ef_search", [16, 64, 256])))
    variants.append(("ivf", {"nlist": ann.get("ivf_nlist", int(4 * np.sqrt(len(vecs))))},
                     "nprobe", ann.get("ivf_nprobe", [4, 16, 64])))

    rows = []
    for spec, params, knob, values in variants:
        t0 = time.perf_counter()
        index = build_faiss_index(vecs, spec, **params)
        build_s = time.perf_counter() - t0
        size_mb = faiss.serialize_index(index).nbytes / 1e6
        for value in values:
            if knob:
                set_search_param(index, spec, value)
            lat, recalls, per_q = [], [], []
            for qi, q in enumerate(ws.questions):
                times = []
                for _ in range(repeats):
                    t = time.perf_counter()
                    _, ids = index.search(qvecs[qi : qi + 1], k)
                    times.append(time.perf_counter() - t)
                lat.append(float(np.median(times)))
                got = [int(i) for i in ids[0] if i >= 0]
                recalls.append(len(set(got) & set(exact_ids[qi].tolist())) / k)
                per_q.append(score_question(got, chunks, q, (k,)))
            agg = aggregate(per_q)
            rows.append({
                "index": spec,
                "params": {**params, **({knob: value} if knob else {})},
                f"ann_recall@{k}": round(float(np.mean(recalls)), 4),
                f"hit@{k}": round(agg[f"hit@{k}"], 4),
                **{kk: round(v, 3) for kk, v in latency_summary(lat).items()},
                "build_s": round(build_s, 2),
                "size_mb": round(size_mb, 1),
            })
            print(rows[-1], flush=True)
    report = {"meta": {**_meta(cfg, ws), "chunking": [size, overlap], "n_vectors": len(vecs),
                       "dim": int(vecs.shape[1])}, "rows": rows}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ann.json").write_text(json.dumps(report, indent=1), encoding="utf8")
    return report


def _meta(cfg: BenchConfig, ws: Workspace) -> dict:
    return {
        "config": asdict(cfg),
        "n_filings": len(ws.docs),
        "n_pages": int(sum(len(p) for p in ws.docs.values())),
        "n_questions": len(ws.questions),
        "skipped_questions": ws.skipped,
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def question_type_breakdown(report: dict, key: str) -> dict[str, float]:
    hits = report["per_question_hits"][key]
    by: dict[str, list[float]] = defaultdict(list)
    for qid, h in hits.items():
        by[report["question_types"][qid]].append(h)
    return {t: round(float(np.mean(v)), 4) for t, v in sorted(by.items())}


def _row(r: dict) -> str:
    return (f"| {r['chunk_size']} / {r['overlap']} | {r['method']} | {r['filter']} | "
            f"{r['hit@1']:.3f} | {r['hit@5']:.3f} | {r['hit@10']:.3f} | {r['recall@10']:.3f} | "
            f"{r['mrr@10']:.3f} | {r['doc_hit@10']:.3f} | {r['p50_ms']:.1f} |")


def write_markdown(bench: dict, ann: dict | None, path: str | Path) -> None:
    m = bench["meta"]
    lines = [
        "# Results",
        "",
        f"Generated {m['timestamp_utc']} from commit `{m['git_sha']}` on {m['cpu_count']} CPUs, "
        f"Python {m['python']}, search pinned to one thread.",
        "",
        f"Corpus: {m['n_filings']} filings, {m['n_pages']:,} pages. Questions: {m['n_questions']} "
        f"answerable of 150 ({len(m['skipped_questions'])} skipped because an evidence filing "
        "could not be downloaded or parsed). Dense embedder: "
        f"`{m['config']['embedder']}`.",
        "",
        "## Metadata filter",
        "",
        "| Level | Questions with a company match | Gold filing kept | Mean candidate filings |",
        "|---|---|---|---|",
    ]
    for f in bench["filters"]:
        lines.append(f"| {f['level']} | {f['questions_with_company_match']:.3f} | "
                     f"{f['gold_filing_retained']:.3f} | {f['mean_candidate_filings']:.1f} |")
    lines += [
        "",
        "## Retrieval grid",
        "",
        "| Chunk words / overlap | Method | Filter | Hit@1 | Hit@5 | Hit@10 | Recall@10 | MRR@10 "
        "| Doc hit@10 | p50 ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [_row(r) for r in bench["rows"]]
    best = max(bench["rows"], key=lambda r: (r["hit@10"], r["mrr@10"]))
    key = f"{best['chunk_size']}/{best['overlap']}/{best['method']}/{best['filter']}"
    lines += [
        "",
        f"Best by Hit@10: `{key}`. Hit@10 by question type for that configuration:",
        "",
        "| Question type | Hit@10 |",
        "|---|---|",
    ]
    lines += [f"| {t} | {v:.3f} |" for t, v in question_type_breakdown(bench, key).items()]
    if ann:
        am = ann["meta"]
        lines += [
            "",
            "## Exact vs approximate search",
            "",
            f"{am['n_vectors']:,} vectors of dimension {am['dim']} (chunking "
            f"{am['chunking'][0]} / {am['chunking'][1]}). Latency is the median of 5 runs per "
            "query, single thread.",
            "",
            "| Index | Params | ANN recall@10 vs exact | Hit@10 | p50 ms | p95 ms "
            "| Build s | Size MB |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in ann["rows"]:
            params = ", ".join(f"{k}={v}" for k, v in r["params"].items()) or "-"
            lines.append(f"| {r['index']} | {params} | {r['ann_recall@10']:.3f} | "
                         f"{r['hit@10']:.3f} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | "
                         f"{r['build_s']:.2f} | {r['size_mb']:.1f} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf8")
