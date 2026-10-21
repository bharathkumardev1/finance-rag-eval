.PHONY: install test

install:
	pip install -e ".[dev]"

test:
	ruff check src tests
	pytest -q
