"""Measure two alternatives I didn't ship, so the README can say why with numbers.

1. int8 query encoder. Dynamic int8 quantization of MiniLM (onnxruntime.quantization)
   against the fp32 model: query-encoding latency, cosine agreement, and what it does to
   test-split retrieval when page vectors stay fp32.
2. Approximate search. Exact MaxP search over every window vs FAISS HNSW, with and without
   the metadata filter: latency, and how often HNSW returns the exact top 10 pages.

Writes results/alternatives.json. Needs `pip install faiss-cpu` for part 2.

    python scripts/alternatives.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from finrag import evaluate as ev  # noqa: E402
from finrag.data import load_questions  # noqa: E402
from finrag.dense import MiniLM  # noqa: E402
from finrag.search import Options, Searcher, load_aliases, top_k  # noqa: E402

DATA = Path("data")
MODEL = Path("models/all-MiniLM-L6-v2")
OUT = Path("results/alternatives.json")


def pct(xs, q):
    return float(np.percentile(np.asarray(xs) * 1000, q))


def int8_encoder(questions, searcher, split) -> dict:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    q8 = MODEL / "onnx" / "model_int8.onnx"
    if not q8.exists():
        quantize_dynamic(str(MODEL / "onnx" / "model.onnx"), str(q8), weight_type=QuantType.QInt8)
    fp32 = MiniLM(MODEL, threads=1)
    int8 = MiniLM(MODEL, threads=1, onnx_file="onnx/model_int8.onnx", verify=False)
    texts = [q.text for q in questions]
    for m in (fp32, int8):  # warm up
        m.encode(texts[:5])
    timings = {"fp32": [], "int8": []}
    vecs = {"fp32": [], "int8": []}
    for t in texts:
        for name, m in (("fp32", fp32), ("int8", int8)):
            t0 = time.perf_counter()
            v = m.encode([t])[0]
            timings[name].append(time.perf_counter() - t0)
            vecs[name].append(v)
    a, b = np.stack(vecs["fp32"]), np.stack(vecs["int8"])
    cos = (a * b).sum(1)
    # Retrieval with int8 query vectors against the fp32 page vectors, full pipeline.
    opts = Options(drop_company_terms=False)
    out = {}
    for name, qv in (("fp32", a), ("int8", b)):
        recs = []
        for q, v in zip(questions, qv, strict=True):
            if q.qid not in split["test"]:
                continue
            _, rows = searcher.filtered(q.text, opts.filter, opts.quarters)
            ranked, _ = searcher.rank(q.text, "dense", rows, 100, v, opts)
            recs.append(ev.score_ranking(ranked.tolist(), searcher, q))
        out[name] = ev.metrics(recs)
    size = {n: (MODEL / "onnx" / f).stat().st_size / 2**20
            for n, f in (("fp32", "model.onnx"), ("int8", "model_int8.onnx"))}
    return {
        "encode_p50_ms": {n: pct(v, 50) for n, v in timings.items()},
        "encode_p95_ms": {n: pct(v, 95) for n, v in timings.items()},
        "model_mb": size,
        "cosine_fp32_vs_int8": {"mean": float(cos.mean()), "min": float(cos.min())},
        "test_dense_full": out,
    }


def ann(questions, searcher) -> dict:
    import faiss

    faiss.omp_set_num_threads(1)
    d = searcher.dense
    vecs = d.vectors
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(vecs.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(vecs)
    build_s = time.perf_counter() - t0
    qv = searcher.encoder.encode([q.text for q in questions])
    res = {"windows": int(len(vecs)), "hnsw_build_s": build_s,
           "hnsw_mb": faiss.serialize_index(index).nbytes / 2**20,
           "flat_mb": vecs.nbytes / 2**20}

    def exact_pages(v):
        return top_k(d.scores(v), 10).tolist()

    exact_t = []
    for v in qv:
        t0 = time.perf_counter()
        exact_pages(v)
        exact_t.append(time.perf_counter() - t0)
    res["exact_unfiltered_p50_ms"] = pct(exact_t, 50)
    res["exact_unfiltered_p95_ms"] = pct(exact_t, 95)
    for ef in (64, 256):
        index.hnsw.efSearch = ef
        times, agree = [], []
        for v in qv:
            t0 = time.perf_counter()
            _, ids = index.search(v[None, :], 400)
            pages, seen = [], set()
            for w in ids[0].tolist():
                if w < 0:
                    continue
                p = int(d.window_page[w])
                if p not in seen:
                    seen.add(p)
                    pages.append(p)
                if len(pages) == 10:
                    break
            times.append(time.perf_counter() - t0)
            agree.append(len(set(pages) & set(exact_pages(v))) / 10)
        res[f"hnsw_ef{ef}"] = {"p50_ms": pct(times, 50), "p95_ms": pct(times, 95),
                               "top10_page_overlap_with_exact": float(np.mean(agree))}
    filt = []
    for q, v in zip(questions, qv, strict=True):
        _, rows = searcher.filtered(q.text, "company_year")
        if rows is None:
            continue
        t0 = time.perf_counter()
        top_k(d.scores(v, rows), 10)
        filt.append(time.perf_counter() - t0)
    res["exact_filtered_p50_ms"] = pct(filt, 50)
    res["exact_filtered_p95_ms"] = pct(filt, 95)
    return res


def main() -> None:
    searcher = Searcher.build(DATA, load_aliases(Path("configs/aliases.yaml")), MODEL,
                              DATA / "emb" / "minilm")
    questions = load_questions(DATA)
    split = json.loads(Path("configs/split.json").read_text())
    out = {"int8_query_encoder": int8_encoder(questions, searcher, split)}
    try:
        out["ann"] = ann(questions, searcher)
    except ImportError:
        out["ann"] = "faiss-cpu not installed"
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
