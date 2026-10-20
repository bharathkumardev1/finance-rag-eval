.PHONY: install data embed eval alternatives loadtest test gate

install:
	pip install -e ".[dev]"

# 360 filing PDFs (about 670 MB), page text, and a check that labeled pages line up.
data:
	finrag download
	finrag pages
	finrag verify

# MiniLM over every page: about 3.5 hours on 2 cores, incremental if interrupted.
embed:
	finrag model
	finrag embed

# Every configuration, both splits: results/eval.json and results/RESULTS.md.
eval:
	finrag eval

# int8 query encoder and HNSW, measured against what ships: results/alternatives.json.
alternatives:
	pip install faiss-cpu
	python scripts/alternatives.py

# Service on one core, load generator on the other: results/loadtest.json.
loadtest:
	bash scripts/loadtest.sh

test:
	ruff check src tests scripts
	pytest -q
	cd web && npm test

gate:
	finrag eval --data-dir tests/fixtures/mini --split tests/fixtures/mini/split.json --aliases none --out ci-results --no-dense
	finrag gate --results ci-results/eval.json --baseline configs/gate.json
