# Results

Produced by `finrag eval` at commit `8034887` on 2026-10-21. FinanceBench at `cc39aeb`: 360 filings, 53,399 pages, 171,158 MiniLM windows, 150 questions (50 dev, 100 test). Intel(R) Xeon(R) Processor @ 2.10GHz, 2 CPUs, Python 3.13.16, numpy 2.5.3. Latency is in-process wall time per query, one query at a time on one thread, including MiniLM query encoding for dense and hybrid.

Headline system, picked on dev: `hybrid/company_year+quarter+drop+route`.

## Test (100 questions)

| Config | Hit@1 | Hit@5 | Hit@10 (95% CI) | MRR@10 | Recall@10 | Right filing@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| `bm25/all-pages` | 0.07 | 0.12 | 0.14 (0.08 to 0.21) | 0.091 | 0.14 | 0.42 | 2.7 | 5.3 |
| `bm25/company` | 0.07 | 0.14 | 0.17 (0.10 to 0.25) | 0.100 | 0.17 | 0.76 | 2.5 | 5.2 |
| `bm25/company_year` | 0.09 | 0.21 | 0.41 (0.32 to 0.51) | 0.155 | 0.39 | 0.93 | 2.3 | 5.0 |
| `bm25/company_year+quarter` | 0.09 | 0.21 | 0.43 (0.34 to 0.53) | 0.158 | 0.41 | 0.94 | 2.2 | 5.1 |
| `bm25/company_year+quarter+route` | 0.40 | 0.61 | 0.71 (0.62 to 0.80) | 0.499 | 0.69 | 0.93 | 2.5 | 5.1 |
| `bm25/company_year+quarter+drop+route` | 0.43 | 0.67 | 0.75 (0.66 to 0.83) | 0.534 | 0.72 | 0.92 | 2.4 | 4.8 |
| `bm25/company_year+quarter+drop` | 0.12 | 0.25 | 0.33 (0.24 to 0.42) | 0.183 | 0.32 | 0.92 | 2.2 | 4.9 |
| `bm25/company_year+drop+route` | 0.43 | 0.66 | 0.74 (0.65 to 0.82) | 0.529 | 0.71 | 0.92 | 2.4 | 5.1 |
| `bm25/company+drop+route` | 0.13 | 0.40 | 0.61 (0.52 to 0.70) | 0.242 | 0.54 | 0.83 | 2.7 | 5.8 |
| `dense/all-pages` | 0.08 | 0.15 | 0.24 (0.16 to 0.33) | 0.120 | 0.22 | 0.61 | 33.9 | 47.7 |
| `dense/company` | 0.10 | 0.19 | 0.28 (0.19 to 0.37) | 0.145 | 0.26 | 0.76 | 13.5 | 29.7 |
| `dense/company_year` | 0.18 | 0.40 | 0.53 (0.43 to 0.63) | 0.280 | 0.49 | 0.94 | 11.8 | 26.0 |
| `dense/company_year+quarter` | 0.18 | 0.42 | 0.55 (0.45 to 0.65) | 0.285 | 0.51 | 0.95 | 10.8 | 24.7 |
| `dense/company_year+quarter+route` | 0.43 | 0.64 | 0.71 (0.62 to 0.80) | 0.527 | 0.69 | 0.94 | 10.5 | 25.1 |
| `dense/company_year+route` | 0.43 | 0.62 | 0.69 (0.60 to 0.78) | 0.522 | 0.67 | 0.94 | 10.5 | 27.7 |
| `dense/company+route` | 0.22 | 0.42 | 0.55 (0.45 to 0.65) | 0.303 | 0.51 | 0.82 | 13.2 | 30.4 |
| `hybrid/all-pages` | 0.08 | 0.12 | 0.16 (0.09 to 0.24) | 0.101 | 0.15 | 0.55 | 36.4 | 52.7 |
| `hybrid/company` | 0.07 | 0.17 | 0.22 (0.14 to 0.31) | 0.113 | 0.21 | 0.75 | 15.5 | 32.4 |
| `hybrid/company_year` | 0.17 | 0.38 | 0.49 (0.39 to 0.59) | 0.264 | 0.46 | 0.94 | 13.6 | 29.2 |
| `hybrid/company_year+quarter` | 0.17 | 0.39 | 0.50 (0.40 to 0.60) | 0.269 | 0.47 | 0.95 | 13.0 | 30.0 |
| `hybrid/company_year+quarter+route` | 0.43 | 0.63 | 0.69 (0.60 to 0.78) | 0.524 | 0.68 | 0.94 | 12.7 | 30.5 |
| `hybrid/company_year+quarter+drop+route` | 0.45 | 0.66 | 0.73 (0.64 to 0.82) | 0.542 | 0.71 | 0.94 | 12.6 | 30.5 |
| `hybrid/company_year+quarter+drop` | 0.15 | 0.41 | 0.50 (0.40 to 0.60) | 0.253 | 0.46 | 0.95 | 12.6 | 33.2 |
| `hybrid/company_year+drop+route` | 0.45 | 0.66 | 0.72 (0.63 to 0.81) | 0.539 | 0.70 | 0.94 | 12.6 | 33.8 |
| `hybrid/company+drop+route` | 0.18 | 0.42 | 0.55 (0.45 to 0.65) | 0.279 | 0.51 | 0.82 | 14.5 | 34.0 |

25 of the 100 questions have more than one gold page. Recall@10 on those alone:

| Config | Recall@10, multi-page |
|---|---|
| `bm25/all-pages` | 0.02 |
| `bm25/company` | 0.02 |
| `bm25/company_year` | 0.08 |
| `bm25/company_year+quarter` | 0.08 |
| `bm25/company_year+quarter+route` | 0.62 |
| `bm25/company_year+quarter+drop+route` | 0.66 |
| `bm25/company_year+quarter+drop` | 0.04 |
| `bm25/company_year+drop+route` | 0.66 |
| `bm25/company+drop+route` | 0.34 |
| `dense/all-pages` | 0.08 |
| `dense/company` | 0.08 |
| `dense/company_year` | 0.28 |
| `dense/company_year+quarter` | 0.28 |
| `dense/company_year+quarter+route` | 0.64 |
| `dense/company_year+route` | 0.64 |
| `dense/company+route` | 0.38 |
| `hybrid/all-pages` | 0.02 |
| `hybrid/company` | 0.04 |
| `hybrid/company_year` | 0.22 |
| `hybrid/company_year+quarter` | 0.22 |
| `hybrid/company_year+quarter+route` | 0.62 |
| `hybrid/company_year+quarter+drop+route` | 0.64 |
| `hybrid/company_year+quarter+drop` | 0.18 |
| `hybrid/company_year+drop+route` | 0.64 |
| `hybrid/company+drop+route` | 0.34 |

## Paired comparisons on test

Same questions, 10,000 bootstrap resamples of the per-question difference.

| A | B | Metric | A minus B | 95% CI | A better | B better |
|---|---|---|---|---|---|---|
| `hybrid/company_year+quarter+drop+route` | `bm25/all-pages` | hit@10 | +0.590 | +0.480 to +0.700 | 63 | 4 |
| `hybrid/company_year+quarter+drop+route` | `bm25/all-pages` | mrr@10 | +0.451 | +0.352 to +0.547 | 66 | 6 |
| `bm25/company_year+quarter` | `bm25/all-pages` | hit@10 | +0.290 | +0.200 to +0.380 | 30 | 1 |
| `bm25/company_year+quarter` | `bm25/all-pages` | mrr@10 | +0.067 | +0.035 to +0.105 | 32 | 1 |
| `bm25/company_year+quarter+route` | `bm25/company_year+quarter` | hit@10 | +0.280 | +0.190 to +0.370 | 29 | 1 |
| `bm25/company_year+quarter+route` | `bm25/company_year+quarter` | mrr@10 | +0.341 | +0.256 to +0.425 | 46 | 6 |
| `bm25/company_year+quarter+drop+route` | `bm25/company_year+quarter+route` | hit@10 | +0.040 | -0.020 to +0.100 | 7 | 3 |
| `bm25/company_year+quarter+drop+route` | `bm25/company_year+quarter+route` | mrr@10 | +0.034 | -0.015 to +0.087 | 19 | 11 |
| `bm25/company_year+quarter+drop+route` | `bm25/company_year+quarter+drop` | hit@10 | +0.420 | +0.320 to +0.520 | 44 | 2 |
| `bm25/company_year+quarter+drop+route` | `bm25/company_year+quarter+drop` | mrr@10 | +0.351 | +0.264 to +0.439 | 48 | 9 |
| `dense/all-pages` | `bm25/all-pages` | hit@10 | +0.100 | +0.010 to +0.190 | 15 | 5 |
| `dense/all-pages` | `bm25/all-pages` | mrr@10 | +0.029 | -0.015 to +0.076 | 16 | 7 |
| `hybrid/all-pages` | `bm25/all-pages` | hit@10 | +0.020 | -0.030 to +0.070 | 4 | 2 |
| `hybrid/all-pages` | `bm25/all-pages` | mrr@10 | +0.010 | -0.018 to +0.039 | 8 | 4 |
| `dense/company_year+quarter+route` | `bm25/company_year+quarter+drop+route` | hit@10 | -0.040 | -0.130 to +0.050 | 8 | 12 |
| `dense/company_year+quarter+route` | `bm25/company_year+quarter+drop+route` | mrr@10 | -0.007 | -0.070 to +0.053 | 21 | 22 |
| `hybrid/company_year+quarter+drop+route` | `bm25/company_year+quarter+drop+route` | hit@10 | -0.020 | -0.080 to +0.030 | 3 | 5 |
| `hybrid/company_year+quarter+drop+route` | `bm25/company_year+quarter+drop+route` | mrr@10 | +0.008 | -0.034 to +0.048 | 17 | 15 |

## Filter

| Split | Questions | Company recognized | Filtered | Right filing kept | Filings left (mean) |
|---|---|---|---|---|---|
| dev | 50 | 50 | 50 | 49 | 2.4 |
| test | 100 | 97 | 97 | 96 | 2.5 |

## Misses of `hybrid/company_year+quarter+drop+route` on test (27 of 100)

| Category | Count | Examples |
|---|---|---|
| no_company | 3 | financebench_id_00288, financebench_id_00822, financebench_id_00601 |
| filter_dropped_filing | 1 | financebench_id_00651 |
| statement_not_detected | 2 | financebench_id_00499, financebench_id_00591 |
| statement_outranked | 1 | financebench_id_10499 |
| other_filing_won | 4 | financebench_id_01476, financebench_id_02416, financebench_id_00283 |
| vocabulary_gap | 12 | financebench_id_04672, financebench_id_01865, financebench_id_01148 |
| outranked | 4 | financebench_id_01928, financebench_id_01107, financebench_id_02119 |

## Hit@10 by question type on test

| Config | domain-relevant Hit@10 | metrics-generated Hit@10 | novel-generated Hit@10 |
|---|---|---|---|
| `hybrid/company_year+quarter+drop+route` | 0.73 (n=33) | 0.94 (n=33) | 0.53 (n=34) |
| `bm25/all-pages` | 0.09 (n=33) | 0.00 (n=33) | 0.32 (n=34) |

## Dev (50 questions)

| Config | Hit@1 | Hit@5 | Hit@10 (95% CI) | MRR@10 | Recall@10 | Right filing@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| `bm25/all-pages` | 0.04 | 0.08 | 0.10 (0.02 to 0.18) | 0.052 | 0.10 | 0.46 | 2.6 | 5.2 |
| `bm25/company` | 0.06 | 0.08 | 0.12 (0.04 to 0.22) | 0.070 | 0.11 | 0.76 | 2.5 | 5.1 |
| `bm25/company_year` | 0.08 | 0.16 | 0.28 (0.16 to 0.40) | 0.130 | 0.25 | 0.96 | 2.3 | 4.6 |
| `bm25/company_year+quarter` | 0.08 | 0.18 | 0.30 (0.18 to 0.42) | 0.141 | 0.27 | 0.96 | 2.2 | 4.5 |
| `bm25/company_year+quarter+route` | 0.40 | 0.60 | 0.64 (0.50 to 0.78) | 0.499 | 0.63 | 0.96 | 2.4 | 4.8 |
| `bm25/company_year+quarter+drop+route` | 0.38 | 0.64 | 0.66 (0.52 to 0.78) | 0.506 | 0.66 | 0.94 | 2.4 | 5.3 |
| `bm25/company_year+quarter+drop` | 0.10 | 0.22 | 0.32 (0.20 to 0.44) | 0.152 | 0.31 | 0.94 | 2.1 | 4.8 |
| `bm25/company_year+drop+route` | 0.36 | 0.64 | 0.66 (0.52 to 0.78) | 0.491 | 0.66 | 0.94 | 2.3 | 5.1 |
| `bm25/company+drop+route` | 0.12 | 0.38 | 0.64 (0.50 to 0.76) | 0.239 | 0.56 | 0.92 | 2.7 | 6.1 |
| `dense/all-pages` | 0.06 | 0.16 | 0.16 (0.06 to 0.26) | 0.095 | 0.15 | 0.54 | 33.5 | 45.6 |
| `dense/company` | 0.06 | 0.16 | 0.24 (0.14 to 0.36) | 0.106 | 0.21 | 0.88 | 12.3 | 24.3 |
| `dense/company_year` | 0.16 | 0.38 | 0.44 (0.30 to 0.58) | 0.243 | 0.42 | 0.98 | 11.0 | 21.9 |
| `dense/company_year+quarter` | 0.18 | 0.44 | 0.48 (0.34 to 0.62) | 0.271 | 0.46 | 0.98 | 10.5 | 23.1 |
| `dense/company_year+quarter+route` | 0.50 | 0.72 | 0.74 (0.62 to 0.86) | 0.580 | 0.74 | 0.98 | 9.7 | 23.2 |
| `dense/company_year+route` | 0.46 | 0.68 | 0.72 (0.60 to 0.84) | 0.550 | 0.72 | 0.98 | 9.5 | 23.9 |
| `dense/company+route` | 0.12 | 0.42 | 0.58 (0.44 to 0.72) | 0.239 | 0.51 | 0.90 | 11.5 | 27.6 |
| `hybrid/all-pages` | 0.06 | 0.10 | 0.14 (0.06 to 0.24) | 0.083 | 0.14 | 0.58 | 34.1 | 53.2 |
| `hybrid/company` | 0.06 | 0.08 | 0.26 (0.14 to 0.38) | 0.091 | 0.22 | 0.86 | 13.4 | 32.8 |
| `hybrid/company_year` | 0.10 | 0.34 | 0.46 (0.32 to 0.60) | 0.195 | 0.42 | 0.94 | 12.1 | 28.9 |
| `hybrid/company_year+quarter` | 0.12 | 0.38 | 0.52 (0.38 to 0.66) | 0.220 | 0.48 | 0.94 | 11.9 | 28.5 |
| `hybrid/company_year+quarter+route` | 0.48 | 0.64 | 0.70 (0.58 to 0.82) | 0.552 | 0.69 | 0.96 | 11.5 | 26.7 |
| `hybrid/company_year+quarter+drop+route` | 0.54 | 0.66 | 0.78 (0.66 to 0.88) | 0.606 | 0.77 | 0.96 | 11.7 | 28.4 |
| `hybrid/company_year+quarter+drop` | 0.20 | 0.42 | 0.54 (0.40 to 0.68) | 0.289 | 0.50 | 0.96 | 11.5 | 27.7 |
| `hybrid/company_year+drop+route` | 0.50 | 0.66 | 0.74 (0.62 to 0.86) | 0.584 | 0.73 | 0.96 | 11.4 | 27.6 |
| `hybrid/company+drop+route` | 0.10 | 0.38 | 0.60 (0.46 to 0.74) | 0.220 | 0.54 | 0.88 | 13.0 | 29.8 |

## Incremental index

From `finrag check-incremental` at commit `8034887` on 2026-10-21. Every tenth filing (36 filings, 5,263 pages) was held out, then added in batches of 12 with a save and reload after each, leaving 4 segments. Against a one-pass build, over 300 queries (every question as asked and with the company's name dropped): largest score difference 0.0e+00 (tolerance 1e-09), identical top 100 for 300 of 300. Building all 360 filings took 13.1 s; each added batch took 0.64, 0.60, 0.42 s including the reload and save.

## Chunk size (BM25)

From `finrag chunk-study` at commit `8034887` on 2026-10-21. BM25 over chunks that never cross a page, a page scoring as its best chunk, against BM25 over whole pages.

| Unit | Documents indexed |
|---|---|
| pages | 53,399 |
| chunks-128-32 | 206,624 |
| chunks-256-64 | 109,629 |
| chunks-650-100 | 55,128 |

Test:

| Config | Hit@1 | Hit@5 | Hit@10 (95% CI) | MRR@10 | Recall@10 | Right filing@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| `bm25-pages/all-pages` | 0.07 | 0.12 | 0.14 (0.08 to 0.21) | 0.091 | 0.14 | 0.42 | 2.7 | 5.0 |
| `bm25-pages/company_year+quarter+drop+route` | 0.43 | 0.67 | 0.75 (0.66 to 0.83) | 0.534 | 0.72 | 0.92 | 2.7 | 5.2 |
| `bm25-chunks-128-32/all-pages` | 0.06 | 0.10 | 0.11 (0.05 to 0.17) | 0.078 | 0.11 | 0.40 | 7.9 | 13.8 |
| `bm25-chunks-128-32/company_year+quarter+drop+route` | 0.38 | 0.65 | 0.72 (0.63 to 0.81) | 0.497 | 0.70 | 0.93 | 7.5 | 13.3 |
| `bm25-chunks-256-64/all-pages` | 0.05 | 0.11 | 0.13 (0.07 to 0.20) | 0.079 | 0.13 | 0.37 | 5.6 | 9.7 |
| `bm25-chunks-256-64/company_year+quarter+drop+route` | 0.41 | 0.67 | 0.71 (0.62 to 0.80) | 0.519 | 0.69 | 0.92 | 5.3 | 9.3 |
| `bm25-chunks-650-100/all-pages` | 0.07 | 0.12 | 0.14 (0.08 to 0.21) | 0.091 | 0.14 | 0.43 | 3.0 | 5.7 |
| `bm25-chunks-650-100/company_year+quarter+drop+route` | 0.43 | 0.67 | 0.75 (0.66 to 0.83) | 0.533 | 0.72 | 0.92 | 3.0 | 5.7 |

Same questions, 10,000 bootstrap resamples of the per-question difference.

| A | B | Metric | A minus B | 95% CI | A better | B better |
|---|---|---|---|---|---|---|
| `bm25-chunks-128-32/all-pages` | `bm25-pages/all-pages` | hit@10 | -0.030 | -0.080 to +0.010 | 1 | 4 |
| `bm25-chunks-128-32/all-pages` | `bm25-pages/all-pages` | mrr@10 | -0.013 | -0.048 to +0.018 | 4 | 6 |
| `bm25-chunks-128-32/company_year+quarter+drop+route` | `bm25-pages/company_year+quarter+drop+route` | hit@10 | -0.030 | -0.080 to +0.020 | 2 | 5 |
| `bm25-chunks-128-32/company_year+quarter+drop+route` | `bm25-pages/company_year+quarter+drop+route` | mrr@10 | -0.037 | -0.066 to -0.009 | 8 | 21 |
| `bm25-chunks-256-64/all-pages` | `bm25-pages/all-pages` | hit@10 | -0.010 | -0.030 to +0.000 | 0 | 1 |
| `bm25-chunks-256-64/all-pages` | `bm25-pages/all-pages` | mrr@10 | -0.013 | -0.038 to +0.006 | 2 | 4 |
| `bm25-chunks-256-64/company_year+quarter+drop+route` | `bm25-pages/company_year+quarter+drop+route` | hit@10 | -0.040 | -0.080 to -0.010 | 0 | 4 |
| `bm25-chunks-256-64/company_year+quarter+drop+route` | `bm25-pages/company_year+quarter+drop+route` | mrr@10 | -0.015 | -0.035 to +0.001 | 4 | 11 |
| `bm25-chunks-650-100/all-pages` | `bm25-pages/all-pages` | hit@10 | +0.000 | +0.000 to +0.000 | 0 | 0 |
| `bm25-chunks-650-100/all-pages` | `bm25-pages/all-pages` | mrr@10 | +0.000 | +0.000 to +0.000 | 0 | 0 |
| `bm25-chunks-650-100/company_year+quarter+drop+route` | `bm25-pages/company_year+quarter+drop+route` | hit@10 | +0.000 | +0.000 to +0.000 | 0 | 0 |
| `bm25-chunks-650-100/company_year+quarter+drop+route` | `bm25-pages/company_year+quarter+drop+route` | mrr@10 | -0.000 | -0.001 to +0.000 | 0 | 1 |

## Load test

From `finrag loadtest --methods bm25,hybrid --levels 1,2,4,8,16 --seconds 30 --server-note '1 uvicorn worker pinned to CPU 0, load generator on CPU 1'` at commit `8034887` on 2026-10-21. 1 uvicorn worker pinned to CPU 0, load generator on CPU 1. Closed-loop clients, 30 s per level, cycling through the 150 questions with the full pipeline. Cost per million queries prices one vCPU at $0.0446 an hour (half a c7i.large on demand in us-east-1).

| Method | Clients | Queries/s | p50 ms | p95 ms | p99 ms | Errors | Server RSS MB | $ per million |
|---|---|---|---|---|---|---|---|---|
| bm25 | 1 | 125.8 | 7.6 | 11.6 | 14.2 | 0 | 943 | 0.10 |
| bm25 | 2 | 139.6 | 13.9 | 20.8 | 27.0 | 0 | 946 | 0.09 |
| bm25 | 4 | 143.6 | 26.8 | 44.8 | 55.1 | 0 | 952 | 0.09 |
| bm25 | 8 | 154.4 | 49.3 | 85.4 | 105.0 | 0 | 965 | 0.08 |
| bm25 | 16 | 162.4 | 95.3 | 156.4 | 195.4 | 0 | 984 | 0.08 |
| hybrid | 1 | 49.4 | 18.2 | 35.2 | 43.4 | 0 | 985 | 0.25 |
| hybrid | 2 | 51.2 | 35.3 | 69.0 | 85.4 | 0 | 998 | 0.24 |
| hybrid | 4 | 54.0 | 67.8 | 129.0 | 158.7 | 0 | 1023 | 0.23 |
| hybrid | 8 | 55.1 | 135.1 | 246.7 | 324.1 | 0 | 1072 | 0.23 |
| hybrid | 16 | 51.8 | 299.2 | 482.5 | 613.9 | 0 | 1163 | 0.24 |
