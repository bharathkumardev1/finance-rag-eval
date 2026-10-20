.PHONY: install data bench test gate

install:
	pip install -e ".[dev]"

data:
	finrag download
	finrag pages

bench:
	finrag bench --config configs/default.yaml --out results
	finrag ann --config configs/default.yaml --out results
	finrag report --out results

test:
	ruff check src tests
	pytest -q

gate:
	finrag bench --config configs/ci.yaml --out ci-results
	finrag gate --results ci-results/benchmark.json --baseline configs/gate.json
