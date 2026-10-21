.PHONY: install data embed eval studies loadtest lock image test

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

# Every configuration on dev and test: results/eval.json and results/RESULTS.md.
eval:
	finrag eval

# The incremental-index check and the chunk-size study. The reranker and generation runs
# need a Hugging Face download and an OpenAI key; see the README.
studies:
	finrag check-incremental
	finrag chunk-study

# Service on one CPU, load generator on the other: results/load.json.
loadtest:
	bash scripts/loadtest.sh

lock:
	uv pip compile pyproject.toml --extra service -c constraints.txt -o requirements.lock \
	  --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes \
	  --custom-compile-command "make lock"

image:
	docker build -t finrag .

test:
	ruff check src tests
	pytest -q
