.PHONY: install data test

install:
	pip install -e ".[dev]"

# FinanceBench at a pinned commit: 150 questions, filing metadata and 360 PDFs (about 670 MB),
# each checked against configs/financebench.sha256. Then page text (2 minutes on 2 cores) and
# a check that every labeled evidence page matches the extracted page with the same number.
data:
	finrag download
	finrag pages
	finrag verify

test:
	ruff check src tests
	pytest -q
