# Results

Generated 2026-10-20T04:37:12Z from commit `616cd1e` on 2 CPUs, Python 3.13.16, search pinned to one thread.

Corpus: 360 filings, 53,399 pages. Questions: 150 answerable of 150 (0 skipped because an evidence filing could not be downloaded or parsed). Dense embedder: `lsa-256`.

## Metadata filter

| Level | Questions with a company match | Gold filing kept | Mean candidate filings |
|---|---|---|---|
| none | 0.000 | 1.000 | 360.0 |
| company | 0.980 | 1.000 | 16.4 |
| company_year | 0.980 | 0.987 | 9.8 |

## Retrieval grid

| Chunk words / overlap | Method | Filter | Hit@1 | Hit@5 | Hit@10 | Recall@10 | MRR@10 | Doc hit@10 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| 256 / 0 | bm25 | none | 0.060 | 0.093 | 0.120 | 0.120 | 0.075 | 0.393 | 0.9 |
| 256 / 0 | dense | none | 0.020 | 0.040 | 0.060 | 0.060 | 0.029 | 0.287 | 59.1 |
| 256 / 0 | hybrid | none | 0.040 | 0.067 | 0.073 | 0.073 | 0.048 | 0.360 | 52.4 |
| 256 / 0 | bm25 | company | 0.073 | 0.133 | 0.180 | 0.173 | 0.100 | 0.753 | 0.8 |
| 256 / 0 | dense | company | 0.047 | 0.113 | 0.173 | 0.173 | 0.077 | 0.707 | 47.1 |
| 256 / 0 | hybrid | company | 0.080 | 0.133 | 0.167 | 0.160 | 0.102 | 0.713 | 49.7 |
| 256 / 0 | bm25 | company_year | 0.127 | 0.240 | 0.333 | 0.323 | 0.179 | 0.920 | 0.9 |
| 256 / 0 | dense | company_year | 0.113 | 0.233 | 0.353 | 0.340 | 0.173 | 0.927 | 47.3 |
| 256 / 0 | hybrid | company_year | 0.127 | 0.293 | 0.393 | 0.383 | 0.194 | 0.927 | 48.4 |
| 512 / 0 | bm25 | none | 0.047 | 0.127 | 0.153 | 0.153 | 0.082 | 0.440 | 0.6 |
| 512 / 0 | dense | none | 0.007 | 0.033 | 0.080 | 0.073 | 0.023 | 0.380 | 29.9 |
| 512 / 0 | hybrid | none | 0.027 | 0.060 | 0.120 | 0.117 | 0.049 | 0.427 | 29.2 |
| 512 / 0 | bm25 | company | 0.073 | 0.160 | 0.200 | 0.197 | 0.107 | 0.733 | 0.7 |
| 512 / 0 | dense | company | 0.020 | 0.113 | 0.187 | 0.177 | 0.057 | 0.727 | 24.1 |
| 512 / 0 | hybrid | company | 0.040 | 0.153 | 0.220 | 0.210 | 0.090 | 0.740 | 24.3 |
| 512 / 0 | bm25 | company_year | 0.107 | 0.293 | 0.367 | 0.357 | 0.185 | 0.933 | 0.7 |
| 512 / 0 | dense | company_year | 0.073 | 0.293 | 0.360 | 0.342 | 0.163 | 0.913 | 25.3 |
| 512 / 0 | hybrid | company_year | 0.140 | 0.313 | 0.387 | 0.377 | 0.218 | 0.933 | 25.9 |
| 512 / 128 | bm25 | none | 0.053 | 0.100 | 0.127 | 0.127 | 0.075 | 0.407 | 0.8 |
| 512 / 128 | dense | none | 0.027 | 0.060 | 0.080 | 0.073 | 0.042 | 0.367 | 55.0 |
| 512 / 128 | hybrid | none | 0.027 | 0.053 | 0.087 | 0.087 | 0.040 | 0.420 | 55.6 |
| 512 / 128 | bm25 | company | 0.073 | 0.127 | 0.207 | 0.207 | 0.104 | 0.740 | 0.7 |
| 512 / 128 | dense | company | 0.027 | 0.107 | 0.160 | 0.153 | 0.061 | 0.713 | 48.5 |
| 512 / 128 | hybrid | company | 0.053 | 0.120 | 0.213 | 0.207 | 0.086 | 0.707 | 42.9 |
| 512 / 128 | bm25 | company_year | 0.140 | 0.260 | 0.360 | 0.352 | 0.201 | 0.940 | 0.7 |
| 512 / 128 | dense | company_year | 0.093 | 0.260 | 0.360 | 0.343 | 0.169 | 0.933 | 43.5 |
| 512 / 128 | hybrid | company_year | 0.133 | 0.293 | 0.387 | 0.380 | 0.206 | 0.920 | 50.5 |
| 1024 / 0 | bm25 | none | 0.053 | 0.133 | 0.207 | 0.200 | 0.089 | 0.500 | 0.5 |
| 1024 / 0 | dense | none | 0.013 | 0.053 | 0.093 | 0.090 | 0.031 | 0.467 | 18.7 |
| 1024 / 0 | hybrid | none | 0.020 | 0.133 | 0.153 | 0.147 | 0.059 | 0.527 | 18.7 |
| 1024 / 0 | bm25 | company | 0.067 | 0.173 | 0.240 | 0.233 | 0.110 | 0.760 | 0.5 |
| 1024 / 0 | dense | company | 0.033 | 0.113 | 0.180 | 0.177 | 0.068 | 0.753 | 19.2 |
| 1024 / 0 | hybrid | company | 0.053 | 0.173 | 0.213 | 0.207 | 0.100 | 0.753 | 17.2 |
| 1024 / 0 | bm25 | company_year | 0.133 | 0.320 | 0.453 | 0.434 | 0.211 | 0.933 | 0.6 |
| 1024 / 0 | dense | company_year | 0.067 | 0.280 | 0.400 | 0.387 | 0.162 | 0.947 | 18.4 |
| 1024 / 0 | hybrid | company_year | 0.107 | 0.340 | 0.440 | 0.422 | 0.208 | 0.927 | 18.3 |

Best by Hit@10: `1024/0/bm25/company_year`. Hit@10 by question type for that configuration:

| Question type | Hit@10 |
|---|---|
| domain-relevant | 0.400 |
| metrics-generated | 0.460 |
| novel-generated | 0.500 |

## Exact vs approximate search

53,461 vectors of dimension 256 (chunking 512 / 0). Latency is the median of 5 runs per query, single thread.

| Index | Params | ANN recall@10 vs exact | Hit@10 | p50 ms | p95 ms | Build s | Size MB |
|---|---|---|---|---|---|---|---|
| flat | - | 1.000 | 0.080 | 4.620 | 5.506 | 0.03 | 54.7 |
| hnsw | M=32, ef_construction=200, efSearch=16 | 0.741 | 0.073 | 0.020 | 0.040 | 15.30 | 69.3 |
| hnsw | M=32, ef_construction=200, efSearch=64 | 0.881 | 0.080 | 0.067 | 0.121 | 15.30 | 69.3 |
| hnsw | M=32, ef_construction=200, efSearch=256 | 0.969 | 0.073 | 0.285 | 0.398 | 15.30 | 69.3 |
| ivf | nlist=924, nprobe=1 | 0.490 | 0.040 | 0.024 | 0.037 | 2.76 | 56.1 |
| ivf | nlist=924, nprobe=4 | 0.811 | 0.060 | 0.033 | 0.056 | 2.76 | 56.1 |
| ivf | nlist=924, nprobe=16 | 0.935 | 0.067 | 0.073 | 0.127 | 2.76 | 56.1 |
| ivf | nlist=924, nprobe=64 | 0.988 | 0.073 | 0.194 | 0.245 | 2.76 | 56.1 |
