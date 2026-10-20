"""Metadata pre-filtering from the question text.

Analysts ask about a named company and period ("What was 3M's FY2018 capital expenditure?").
The filter reads those from the question, without using any gold labels, and narrows retrieval
to that company's filings for that period. Three levels are benchmarked:

* ``none``: search the whole corpus.
* ``company``: only filings from companies named in the question.
* ``company_year``: additionally only filings whose period matches a year in the question.
  If no filing matches the year, it falls back to the company level.

When no company is recognized, the filter returns ``None`` and search runs unfiltered.
"""

from __future__ import annotations

import re

from .data import DocMeta

_SUFFIXES = {
    "inc", "incorporated", "corporation", "corp", "co", "company", "companies", "group",
    "holdings", "holding", "plc", "ltd", "limited", "the", "and", "lp", "llc", "sa", "nv",
}
_YEAR = re.compile(r"\b(?:fy\s?)?((?:19|20)\d{2})\b")
_SHORT_FY = re.compile(r"\bfy\s?'?(\d{2})\b")

LEVELS = ("none", "company", "company_year")


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


class MetadataFilter:
    def __init__(
        self, metas: dict[str, DocMeta], extra_aliases: dict[str, list[str]] | None = None
    ):
        self.metas = metas
        aliases: dict[str, str] = {}
        for company in sorted({m.company for m in metas.values()}):
            full = _norm(company)
            core = " ".join(t for t in full.split() if t not in _SUFFIXES)
            for alias in {full, core}:
                if len(alias) >= 2:
                    aliases.setdefault(alias, company)
        for company, extra in (extra_aliases or {}).items():
            for alias in extra:
                aliases.setdefault(_norm(alias), company)
        # Longest aliases first, so "american water works" wins over "american".
        self._aliases = sorted(aliases.items(), key=lambda kv: -len(kv[0]))

    def companies_in(self, text: str) -> set[str]:
        padded = f" {_norm(text)} "
        found = set()
        for alias, company in self._aliases:
            if f" {alias} " in padded:
                found.add(company)
                padded = padded.replace(f" {alias} ", " ")
        return found

    @staticmethod
    def years_in(text: str) -> set[int]:
        low = text.lower()
        years = {int(y) for y in _YEAR.findall(low)}
        years |= {2000 + int(y) for y in _SHORT_FY.findall(low)}
        return years

    def allowed_docs(self, text: str, level: str) -> set[str] | None:
        if level == "none":
            return None
        companies = self.companies_in(text)
        if not companies:
            return None
        docs = {d for d, m in self.metas.items() if m.company in companies}
        if level == "company_year":
            years = self.years_in(text)
            narrowed = {d for d in docs if self.metas[d].period in years}
            if narrowed:
                docs = narrowed
        return docs
