# finance-rag-eval

Which page of an SEC filing answers an analyst's question, and how often can a retriever put it in the top 10? On FinanceBench's 150 public questions over 360 filings (53,399 pages), BM25 across every page finds it for 14 of the 100 test questions. Reading the company and period from the question and searching only those filings takes that to 43. Sending questions about a balance sheet, income statement or cash flow statement to those pages takes it to 71, and leaving the company's name out of the query to 75, at 2.4 ms a query on one thread.

MiniLM embeddings and a BM25 + MiniLM hybrid end up in the same place once the filter and routing are in, at four to five times the latency:

| Test, 100 questions | Hit@1 | Hit@10 (95% CI) | MRR@10 | p50 / p95 ms |
|---|---|---|---|---|
| BM25, all pages | 0.07 | 0.14 (0.08 to 0.21) | 0.091 | 2.7 / 5.3 |
| MiniLM, all pages | 0.08 | 0.24 (0.16 to 0.33) | 0.120 | 33.9 / 47.7 |
| Hybrid, all pages | 0.08 | 0.16 (0.09 to 0.24) | 0.101 | 36.4 / 52.7 |
| BM25 + filter | 0.09 | 0.43 (0.34 to 0.53) | 0.158 | 2.2 / 5.1 |
| BM25 + filter + routing | 0.40 | 0.71 (0.62 to 0.80) | 0.499 | 2.5 / 5.1 |
| BM25 + filter + routing, name dropped | 0.43 | 0.75 (0.66 to 0.83) | 0.534 | 2.4 / 4.8 |
| MiniLM + filter + routing | 0.43 | 0.71 (0.62 to 0.80) | 0.527 | 10.5 / 25.1 |
| Hybrid + filter + routing, name dropped | 0.45 | 0.73 (0.64 to 0.82) | 0.542 | 12.6 / 30.5 |

The last row is the reported system because it won on the 50 dev questions (0.78 against 0.66 for the same pipeline on BM25 alone). On test the two tie: 0.73 against 0.75, a difference of -0.02 (95% CI -0.08 to +0.03). On these questions the embeddings don't buy anything BM25 doesn't, and BM25 costs a fifth of the latency.

Paired on the same questions with 10,000 bootstrap resamples, the filter adds 0.29 to BM25 alone (0.20 to 0.38) and routing adds 0.28 on top of the filter (0.19 to 0.37). Dropping the name adds 0.04 (-0.02 to +0.10), which is within noise. Latency is in-process wall time for one query at a time on one thread of a 2.1 GHz Xeon, MiniLM query encoding included; [results/RESULTS.md](results/RESULTS.md) has every configuration, the dev split and the ablations.

## The pipeline

- Filter. The question names a company and usually a period ("What was 3M's FY2018 capital expenditure?"). The filter reads them from the question text, never from the labels, and keeps that company's filings for that year, or that quarter's 10-Q when a quarter is named. Each step falls back to the one before when nothing matches. It keeps the filing with the answer for 96 of the 100 test questions and 49 of the 50 dev questions. Company aliases came from the dev split only.
- Routing. For 78 of the 150 questions, the answer is on a primary financial statement. Those pages are tables, and they share almost no words with "what is the FY2019 fixed asset turnover ratio", so BM25 ranks them far down. Routing recognizes statement pages by their titles, reads which statement a question needs from the statement it names or a standard line item or ratio ("working capital", "capex", "effective tax rate"), and moves pages of that kind to the front. I wrote the patterns from accounting definitions and checked them on dev only.
- BM25 is a Lucene-style scorer in float64 over pages, kept as segments, so adding filings doesn't rebuild the index. MiniLM (all-MiniLM-L6-v2 through ONNX Runtime) embeds each page as 254-token windows, 171,158 of them, and a page scores as its best window. The hybrid fuses the two rankings with reciprocal rank fusion (k = 60, depth 100).

## Where it still misses

The reported system misses 27 of the 100 test questions. Each miss goes in the first stage that lost the gold page, computed rather than hand-labeled: 12 are a vocabulary gap (a narrative page sharing at most two words with the question once the company's name is gone), 4 lost to pages from another filing the filter kept, 4 lost to pages sharing more words, 3 named no company the filter knows, 2 needed a statement nothing in the question pointed to, 1 had the right statement outranked and 1 had its filing dropped by the filter.

By FinanceBench's question types, Hit@10 is 0.94 for the questions generated from standard financial metrics, 0.73 for the standard questions asked of every company, and 0.53 for the company-specific questions financial analysts wrote. 25 test questions need more than one page; the reported system gets 64% of their gold pages into the top 10, and BM25 with the same pipeline 66%.

## Smaller checks

- Adding filings to the index. Every tenth filing (36 filings, 5,263 pages) was held out and added back in batches of 12, saving and reloading each time. Against a one-pass build, the largest score difference over 300 queries was 0.0, and every top 100 was identical. A full build takes 13.1 s; each batch took 0.42 to 0.64 s.
- Chunk size. BM25 over 650-token chunks scores the same as over pages (0.75), 256-token chunks lose 0.04 (-0.08 to -0.01) and 128-token chunks 0.03 (-0.08 to +0.02), while indexing two to four times as many units. Pages stay.

## Serving

`GET /search?q=...` runs the BM25 pipeline above; `method=hybrid` or `method=dense` uses the embeddings when they're mounted. In a closed-loop load test with one uvicorn worker on one core and the load generator on the other, BM25 serves 125.8 queries a second to one client (p50 7.6 ms, p95 11.6 ms) and reaches 162 a second at 16 clients; the hybrid serves 49.4 a second to one client (p95 35.2 ms). There were no errors, and the server held 0.94 to 1.16 GB. Priced at half an AWS c7i.large on demand, that's $0.10 per million BM25 queries and $0.25 per million hybrid queries, assuming this throughput carries over to that vCPU, which I haven't measured.

```bash
docker build -t finrag .
docker run --rm -p 8000:8000 -v "$PWD/data:/data" -v "$PWD/models:/models:ro" finrag
```

Without the mounted embeddings the service runs BM25 only, and `/healthz` says so. The image is `python:3.12-slim` pinned by digest, with dependencies from a hashed lock.

`web/finrag.mjs` is the BM25 pipeline in JavaScript for the browser demo: filter, routing and scoring in the same float64 arithmetic in the same order, and CI requires the same rankings and scores as Python.

## Running it

```bash
make install
make data      # FinanceBench at commit cc39aeb: 150 questions, 360 PDFs (about 670 MB) checked by SHA-256, page text
make embed     # MiniLM over every page window; hours on a laptop CPU, resumes where it stopped
make eval      # every configuration on dev and test: results/eval.json and RESULTS.md
make studies   # the incremental-index check and the chunk-size study
make loadtest
```

Two more studies need a download or an API key, and read their page rankings from the committed `results/eval.json`, so neither reruns retrieval. Smoke runs come first and write to their own files, so they can't touch the real recording:

```bash
finrag reranker                                              # BAAI/bge-reranker-base, 1.1 GB, checked by SHA-256
finrag rerank-study --limit 3 --out /tmp/rerank-smoke.json   # tokenizer and timing check
finrag rerank-study                                          # headline, BM25 with the filter, BM25 over all pages
pip install -e ".[generate]"                                 # the OpenAI client; the key goes in OPENAI_API_KEY
finrag generate --on dev --limit 3 --record /tmp/smoke.jsonl --out /tmp/smoke.json
finrag generate                                              # four settings on the test split; $3 cap over the whole recording
finrag label-judge && finrag judge-agreement
finrag report                                                # adds both to RESULTS.md
```

The generation study answers each test question from four page sets: the gold pages, the reported system's top 5, BM25's top 5 over all pages, and no pages at all. The gaps between them separate what retrieval costs from what the model gets wrong with the right pages, and what it already knows without any. A stronger model judges each answer against FinanceBench's reference. Every call is recorded under a hash of its request, so a rerun replays it, and `--replay` needs no key. The judge check is 60 of its verdicts, graded by me against the reference answers. I used AI tools to find filings and check calculations, so the grading was research-assisted, not blind to AI input; `results/generation/LABELS.md` lists the rows where the reference or the rubric is debatable.

On the 100 test questions the judge marks the model correct on 83 with the gold pages, 66 with the reported system's top 5, 37 with no pages and 14 with BM25's top 5 over all pages. Retrieval costs 0.17 against the gold pages (95% CI 0.09 to 0.25, paired), and the system's pages add 0.29 over no pages at all (0.16 to 0.41); the model already knows some figures for well-known companies, so the no-pages number is not zero. The cross-encoder reranker (`bge-reranker-base`) made retrieval worse: Hit@10 fell from 0.78 to 0.62 on dev and from 0.73 to 0.63 on test, and reading three windows per page did not fix it (0.57 on test), so the reported system stays without a reranker. The judge agreed with my 60 labels on 59 (kappa 0.95, 95% CI 0.83 to 1.00). All numbers are in `results/RESULTS.md`; the recorded generation run cost $0.46.

On every push, CI runs the tests on Python 3.11 and 3.13 (the MiniLM test against recorded vectors), a regression gate that runs the BM25 pipeline over the 40 real filings behind the dev questions and fails if Hit@10 drops more than one question below its recorded 38 of 50, a smoke test of the image, a secret scan, and the browser port's parity test.

## Limits

- 150 questions is the public sample of FinanceBench. 100 test questions give intervals about 0.18 wide on Hit@10, and paired differences smaller than about 0.05 come out as ties.
- The filter's aliases and the routing patterns were written by reading the dev questions; they'd need the same work for a new set of companies.
- Hit@10 asks whether the right page is retrieved, not whether a model answers correctly from it; that's what the generation study is for.
- The reranker's window is 512 tokens, so a long page is scored on its first part unless `--windows` is raised.
- The judge check is 60 verdicts from one grader, so its agreement interval is wide. Some FinanceBench references are debatable (a Pfizer spin-off that had already closed, a JPMorgan figure that is tangible book value), so accuracy here means agreement with the reference.
- Statement pages are recognized from their titles, so a filing that lays them out differently falls back to plain BM25.

## License

MIT, see [LICENSE](LICENSE).
