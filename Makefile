.PHONY: install data embed eval studies loadtest lock image test demo

install:
	pip install -e ".[dev]"

# FinanceBench at a pinned commit: 150 questions, filing metadata and 360 PDFs (about 670 MB),
# each checked against configs/financebench.sha256. Then page text (2 minutes on 2 cores) and
# a check that every labeled evidence page matches the extracted page with the same number.
data:
	finrag download
	finrag pages
	finrag verify

# MiniLM over every page window. Hours on a laptop CPU; picks up where it stopped.
embed:
	finrag model
	finrag embed

# Every configuration on dev and test: results/eval.json and results/RESULTS.md. One thread,
# so per-query latency means the same thing here as in the one-worker service.
eval:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 finrag eval

# The incremental-index check and the chunk-size study. The reranker and generation runs
# need a Hugging Face download and an OpenAI key; see the README.
studies:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 finrag check-incremental
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 finrag chunk-study

# Service on one CPU, load generator on the other: results/load.json.
loadtest:
	bash scripts/loadtest.sh

lock:
	uv pip compile pyproject.toml --extra service -c constraints.txt -o requirements.lock \
	  --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes \
	  --custom-compile-command "make lock"

image:
	docker build -t finrag .

# Lint, pytest, then the browser port against a fresh export of the fixture corpus.
test:
	ruff check src tests
	pytest -q
	finrag export-demo --data-dir tests/fixtures/mini --split tests/fixtures/mini/split.json \
	  --aliases "" --results "" --out web/test/data
	cd web && npm test

# Filings, metadata, parity cases and summary.json for the site's search demo.
demo:
	finrag export-demo --out build/demo
