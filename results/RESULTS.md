# Results

Produced by `finrag eval` at commit `075017e` on 2026-10-21. FinanceBench at `cc39aeb`: 360 filings, 53,399 pages, 150 questions (50 dev, 100 test). Intel(R) Xeon(R) Processor @ 2.10GHz, 2 CPUs, Python 3.13.16, numpy 2.5.3. Latency is in-process wall time per query, one query at a time.

## Test (100 questions)

| Config | Hit@1 | Hit@5 | Hit@10 (95% CI) | MRR@10 | Recall@10 | Right filing@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| `bm25/all-pages` | 0.07 | 0.12 | 0.14 (0.08 to 0.21) | 0.091 | 0.14 | 0.42 | 1.1 | 2.2 |

25 of the 100 questions have more than one gold page. Recall@10 on those alone:

| Config | Recall@10, multi-page |
|---|---|
| `bm25/all-pages` | 0.02 |

By question type:

| Config | domain-relevant Hit@10 | metrics-generated Hit@10 | novel-generated Hit@10 |
|---|---|---|---|
| `bm25/all-pages` | 0.09 (n=33) | 0.00 (n=33) | 0.32 (n=34) |

## Dev (50 questions)

| Config | Hit@1 | Hit@5 | Hit@10 (95% CI) | MRR@10 | Recall@10 | Right filing@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| `bm25/all-pages` | 0.04 | 0.08 | 0.10 (0.02 to 0.18) | 0.052 | 0.10 | 0.46 | 1.2 | 3.0 |
