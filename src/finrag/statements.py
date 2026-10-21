"""Route statement questions to the financial statement pages.

For 78 of the 150 FinanceBench questions, a gold page is a primary financial statement: the
balance sheet, the income statement or the cash flow statement (as ``page_kinds`` below
recognizes them). Those pages are tables. They share almost no words with "what is the
FY2019 fixed asset turnover ratio", so BM25 ranks them far down.

Two small pieces of domain knowledge go after this:

* ``page_kinds`` recognizes a statement page by its title at the top of the page, skipping
  the pages that only mention a statement (indexes, auditor reports, notes) and pages that
  aren't mostly numbers.
* ``question_kinds`` reads which statement a question needs, either named outright
  ("statement of financial position") or implied by a standard line item or ratio
  ("working capital", "capex", "effective tax rate").

``route`` then moves pages of the requested kinds to the front of a ranking, keeping their
order. I wrote the patterns from accounting definitions and checked them on the dev split
only.
"""

from __future__ import annotations

import re

KINDS = ("balance", "income", "cashflow")

_PREFIX = r"(?:consolidated\s+|condensed\s+|combined\s+)*"
_TITLE = {
    "balance": re.compile(
        _PREFIX + r"(?:balance\s+sheets?|statements?\s+of\s+(?:consolidated\s+)?financial\s+"
        r"(?:position|condition))", re.I),
    "income": re.compile(
        _PREFIX + r"(?:statements?\s+of\s+(?:consolidated\s+)?(?:income|operations|earnings)"
        r"(?:\s*\(loss\))?|income\s+statements?)(?!\s+and\s+comprehensive|\s+data)", re.I),
    "cashflow": re.compile(_PREFIX + r"statements?\s+of\s+(?:consolidated\s+)?cash\s+flows?", re.I),
}
_NOT_A_STATEMENT = re.compile(
    r"report of independent|index to|notes to (?:the )?(?:condensed )?consolidated|item 15|"
    r"selected financial data|schedule i\b|parent", re.I)
_DIGIT = re.compile(r"\d")
HEAD_CHARS = 220
MIN_NUMERIC_SHARE = 0.25

_NAMED = {
    "balance": r"balance sheets?|statements? of financial position|financial position",
    "income": r"income statements?|statements? of (?:income|operations|earnings)|p&l|profit and loss",
    "cashflow": r"cash flow statements?|statements? of cash flows?|cash flows? statements?",
}
_IMPLIED = {
    "balance": (
        r"working capital|quick ratio|current ratio|total assets|pp&e|ppne|"
        r"property,? plant,? and equipment|accounts (?:receivable|payable)|inventor(?:y|ies)|"
        r"return on assets|\broa\b|asset turnover|days (?:sales|payable|inventory)|"
        r"debt[- ]to[- ]equity|shareholders'? equity"),
    "income": (
        r"revenue|net sales|net income|net earnings|operating income|"
        r"operating (?:profit )?margin|gross (?:margin|profit)|\bcogs\b|"
        r"cost of (?:goods sold|sales|revenue)|\beps\b|earnings per share|effective tax rate|"
        r"\bebitd?a?r?\b|interest (?:coverage|expense)|sg&a"),
    "cashflow": (
        r"capital expenditures?|\bcapex\b|operating cash flow|cash from operations|"
        r"cash flows? from operating|free cash flow|dividends? (?:paid|payout)|"
        r"depreciation and amortization|\bd&a\b|repurchases?|buybacks?|investing activities|"
        r"financing activities"),
}
_NAMED_RX = {k: re.compile(v, re.I) for k, v in _NAMED.items()}
_IMPLIED_RX = {k: re.compile(v, re.I) for k, v in _IMPLIED.items()}


def page_kinds(text: str) -> frozenset[str]:
    head = " ".join(text[:HEAD_CHARS].split())
    if _NOT_A_STATEMENT.search(head):
        return frozenset()
    tokens = text.split()
    if not tokens or sum(bool(_DIGIT.search(t)) for t in tokens) / len(tokens) < MIN_NUMERIC_SHARE:
        return frozenset()
    return frozenset(k for k, rx in _TITLE.items() if rx.search(head))


def question_kinds(question: str, implied: bool = True) -> frozenset[str]:
    found = {k for k, rx in _NAMED_RX.items() if rx.search(question)}
    if implied:
        found |= {k for k, rx in _IMPLIED_RX.items() if rx.search(question)}
    return frozenset(found)


def route(ranking: list[int], kinds_of_row, wanted: frozenset[str]) -> list[int]:
    """Pages whose kinds intersect ``wanted`` first, each group in its original order."""
    if not wanted:
        return ranking
    first = [r for r in ranking if kinds_of_row[r] & wanted]
    rest = [r for r in ranking if not kinds_of_row[r] & wanted]
    return first + rest
