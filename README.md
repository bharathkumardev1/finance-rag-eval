# finance-rag-eval

[![CI](https://github.com/bharathkumardev1/finance-rag-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/bharathkumardev1/finance-rag-eval/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A retrieval benchmark for RAG over real SEC filings. It measures which design choices actually
move retrieval quality on financial documents: chunk size, BM25 vs dense vs hybrid search,
metadata filtering, and exact vs approximate FAISS indexes.

It runs on [FinanceBench](https://github.com/patronus-ai/financebench) (Islam et al., 2023):
150 analyst questions over 360 public 10-K, 10-Q, 8-K, and earnings filings (53,399 pages).
Every question names the filing and the exact page that answers it, so retrieval can be scored
at the page level instead of by eyeballing answers.

## What the data says

All numbers come from [`results/benchmark.json`](results/benchmark.json) and
[`results/ann.json`](results/ann.json), produced by `make bench` at commit `616cd1e`.
The full grid is in [`results/RESULTS.md`](results/RESULTS.md). The dense embedder in this run
is LSA (TF-IDF plus SVD, 256 dimensions) trained on the filings, because it runs fully offline;
transformer embedders are one config line away (see below).

**1. Finding the right filing is the first problem.** With all 360 filings in one index, the
best configuration (BM25, 1,024-word chunks) puts the evidence page in its top 10 for 20.7% of
questions, and even the right filing shows up in the top 10 for only 50.0%. Financial filings
repeat the same vocabulary across companies and years, so "capital expenditure" matches every
10-K in the corpus.

**2. A metadata filter read from the question fixes most of that.** The filter pulls company
names, tickers, and fiscal years out of the question text, with no access to the gold labels.
It recognized a company in 98.0% of questions, kept the gold filing for 98.7%, and cut the
search space from 360 filings to 9.8 on average. Right-filing-in-top-10 went from 50.0% to
93.3%, and page-level Hit@10 more than doubled, from 20.7% to 45.3%.

**3. After the filter, finding the page is the bottleneck.** The right filing is in the top 10
for 93% of questions, but the right page for only 45%. Questions like "What is the FY2019 fixed
asset turnover ratio for Activision Blizzard?" name a ratio that filings rarely print; the
inputs sit in a balance sheet and an income statement that share few words with the question.

| Filter | Best setup | Hit@10 | Doc hit@10 | MRR@10 |
|---|---|---|---|---|
| none | BM25, 1,024 words | 0.207 | 0.500 | 0.089 |
| company | BM25, 1,024 words | 0.240 | 0.760 | 0.110 |
| company and year | BM25, 1,024 words | **0.453** | **0.933** | **0.211** |

**4. BM25 ranks better than the LSA dense baseline.** It has the higher MRR@10 in all 12
chunk-size and filter settings, although once the filter is on, Hit@10 for the two is within
noise. Hybrid search (reciprocal rank fusion) edges out BM25 at 256 and 512 words (0.387 vs
0.367 Hit@10 at 512 with the filter), but not at 1,024. With 150 questions, one question is
0.0067, and the 95% interval on a Hit@10 near 0.45 is about ±0.08, so differences of a few
points here are noise. The large effect is the filter.

**5. Bigger chunks found the page more often:** 0.333, 0.367, and 0.453 Hit@10 at 256, 512,
and 1,024 words (BM25, filtered). The cost is more text per retrieved chunk in the LLM
context. A 128-word overlap at 512 words changed nothing (0.360 vs 0.367).

**6. At this scale, the index type matters far less than the filter.** On 53,461 vectors, exact
search takes 4.6 ms per query (p50, one thread). HNSW (M=32, efSearch=64) returns 88.1% of the
exact top 10 in 0.067 ms, and IVF (nprobe=16) returns 93.5% in 0.073 ms.

| Index | Setting | Recall vs exact top 10 | p50 ms | p95 ms | Build s | Size MB |
|---|---|---|---|---|---|---|
| Flat (exact) | | 1.000 | 4.620 | 5.506 | 0.03 | 54.7 |
| HNSW | efSearch=64 | 0.881 | 0.067 | 0.121 | 15.3 | 69.3 |
| HNSW | efSearch=256 | 0.969 | 0.285 | 0.398 | 15.3 | 69.3 |
| IVF | nprobe=16 | 0.935 | 0.073 | 0.127 | 2.76 | 56.1 |
| IVF | nprobe=64 | 0.988 | 0.194 | 0.245 | 2.76 | 56.1 |

These results line up with the FinanceBench paper, which found that GPT-4-Turbo with a
retrieval system answered incorrectly or refused 81% of questions.

## How it works

```
filings (PDF) ─► page text ─► page-aware chunks ─► BM25 index ─┐
                                                └► embeddings ─► FAISS ─┤
question ─► metadata filter (company, year) ─► mask ──────────────────┼─► ranked chunks ─► page-level metrics
                                                                       └── hybrid: reciprocal rank fusion
```

* **Page-aware chunking** ([`chunking.py`](src/finrag/chunking.py)). Chunks are word windows that
  remember which PDF pages they span, so a chunk counts as relevant only if it covers an evidence
  page of the right filing. Extraction is checked against the dataset: all 189 evidence pages
  match the extracted page text.
* **Retrievers** ([`retrievers.py`](src/finrag/retrievers.py)). BM25 via `bm25s`, dense search
  via FAISS inner product on unit vectors, and hybrid via reciprocal rank fusion
  (k=60, depth 100). Filtered dense search is exact over the allowed chunks.
* **Metadata filter** ([`filters.py`](src/finrag/filters.py)). Company names, a short ticker
  and nickname table, and years are matched in the question text. If no filing matches the
  year, it falls back to all of the company's filings. If no company is found, search is
  unfiltered.
* **Metrics** ([`metrics.py`](src/finrag/metrics.py)). Hit@k, Recall@k over evidence pages,
  MRR@10, and doc hit@k (right filing, any page).
* **Incremental index** ([`store.py`](src/finrag/store.py)). A FAISS `IndexIDMap2` with a
  manifest of each document's SHA-256. Upserting re-embeds only new or changed filings, removes a
  changed filing's old chunks, and leaves the rest alone. Every chunk id maps back to its filing
  and pages.
* **Quality gate** ([`gate.py`](src/finrag/gate.py)). CI runs the pipeline on a small synthetic
  corpus and fails the build if any tracked metric drops below its baseline.

## Reproduce

```bash
git clone https://github.com/bharathkumardev1/finance-rag-eval && cd finance-rag-eval
pip install -e ".[dev]"
make data    # downloads the 360 filings (~670 MB) and extracts page text
make bench   # retrieval grid, ANN comparison, results/RESULTS.md
```

On a 2-CPU machine the grid takes about 7.5 minutes and the ANN comparison about 1.5. Results
are deterministic: LSA uses a fixed seed, and a rerun matched every metric exactly.
`pip install pymupdf` makes PDF extraction much faster than the default `pypdf`; it is optional
because it is AGPL-licensed.

**With a transformer embedder.** Set `embedder` in `configs/default.yaml` to
`st:BAAI/bge-small-en-v1.5` (after `pip install -e ".[st]"`) or
`openai:text-embedding-3-small` (after `pip install -e ".[openai]"` and setting
`OPENAI_API_KEY`), then run `make bench`.

**Tests.** `make test` runs ruff and 15 tests: chunking, metrics, filters, all three
retrievers, exact vs approximate indexes, the incremental index, and an end-to-end benchmark
plus quality gate on the synthetic fixture. CI runs them on Python 3.10 and 3.12.

## Limits

* The dense results use LSA, a classical baseline. Modern transformer embedders should do better
  on dense and hybrid, and that run is the obvious next step.
* 150 questions is a small sample. Treat differences under about 8 points as noise.
* Relevance is page-level. A chunk that covers the evidence page counts as a hit even if the
  answer needs a second page from another statement.
* The nickname part of the alias table (AMEX, JnJ, JPM, MGM) was written after seeing which
  questions use those names. That is a small amount of tuning on the test questions, and it
  only affects the filter's recognition rate, not the gold labels.
* This measures retrieval only. Answer quality, such as faithfulness scored by an LLM judge,
  is not evaluated yet.

## Layout

```
src/finrag/
  data.py        FinanceBench download, PDF page extraction, typed loaders
  chunking.py    page-aware chunking
  embedders.py   LSA, hashing, sentence-transformers, OpenAI behind one interface
  retrievers.py  BM25, dense (FAISS), hybrid (RRF), metadata masks
  filters.py     company and year filter from question text
  metrics.py     page-level Hit@k, Recall@k, MRR, doc hit@k
  store.py       incremental index keyed by content hash
  bench.py       retrieval grid, ANN comparison, RESULTS.md writer
  gate.py        quality gate used by CI
  cli.py         finrag download | pages | bench | ann | report | gate
configs/         default (full run), ci (fixture), gate baseline
results/         benchmark.json, ann.json, RESULTS.md from the run above
tests/           unit and pipeline tests; synthetic fixture corpus
```

## Data and citation

FinanceBench is published by Patronus AI. This repository does not redistribute the filings or
questions; `finrag download` fetches them from the FinanceBench repository.

Islam, P., Kannappan, A., Kiela, D., Qian, R., Scherrer, N., and Vidgen, B. (2023).
*FinanceBench: A New Benchmark for Financial Question Answering.* arXiv:2311.11944.

## License

MIT. See [LICENSE](LICENSE).
