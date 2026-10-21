.PHONY: install data test

install:
	pip install -e ".[dev]"

# FinanceBench at a pinned commit: 150 questions, filing metadata and 360 PDFs (about 670 MB),
# each checked against configs/financebench.sha256.
data:
	finrag download

test:
	ruff check src tests
	pytest -q
