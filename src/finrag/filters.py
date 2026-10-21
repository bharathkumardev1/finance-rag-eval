"""Narrow the search to the filings a question is about, read from the question text.

Analysts ask about a named company and period ("What was 3M's FY2018 capital expenditure?").
The filter reads those from the question, never from the labels, and keeps only that
company's filings for that period. Levels:

* ``none``: the whole corpus.
* ``company``: filings of the companies named in the question.
* ``company_year``: of those, the filings for a year named in the question. A quarter
  ("2022 Q2", "Q2 of FY2023", "second fiscal quarter of 2023") narrows further to that
  quarter's 10-Q or earnings release when there is one. Each step falls back to the one
  before it when nothing matches.

When no company is recognized the filter returns None and the search runs unfiltered.
"""

from __future__ import annotations

import re
from pathlib import Path

from .data import DocMeta

_SUFFIXES = {
    "inc", "incorporated", "corporation", "corp", "co", "company", "companies", "group",
    "holdings", "holding", "plc", "ltd", "limited", "the", "and", "lp", "llc", "sa", "nv",
}
_YEAR = re.compile(r"\b(?:fy\s?)?((?:19|20)\d{2})\b")
_SHORT_FY = re.compile(r"\bfy\s?'?(\d{2})\b")
_ORDINAL = {"first": 1, "second": 2, "third": 3, "fourth": 4, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
_FY = r"(?:fy\s*|fiscal\s+(?:year\s+)?)?"
_QUARTER = (
    re.compile(r"\bq([1-4])\s*(?:of\s+)?" + _FY + r"'?((?:20)?\d{2})\b", re.I),
    re.compile(r"\b(20\d{2})\s*q([1-4])\b", re.I),
    re.compile(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+(?:fiscal\s+)?quarter\s+(?:of\s+)?"
               + _FY + r"(20\d{2})\b", re.I),
)
_DOC_QUARTER = re.compile(r"_(\d{4})Q([1-4])_")

LEVELS = ("none", "company", "company_year")


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


class MetadataFilter:
    def __init__(self, metas: dict[str, DocMeta], extra_aliases: dict[str, list[str]] | None = None):
        self.metas = metas
        aliases: dict[str, str] = {}
        for company in sorted({m.company for m in metas.values()}):
            full = _norm(company)
            core = " ".join(t for t in full.split() if t not in _SUFFIXES)
            for alias in {full, core}:
                if len(alias) >= 2:
                    aliases.setdefault(alias, company)
        known = {m.company for m in metas.values()}
        for company, extra in (extra_aliases or {}).items():
            if company not in known:  # an alias for a company with no filings would empty the filter
                continue
            for alias in extra:
                aliases.setdefault(_norm(alias), company)
        # Longest first, so "american water works" wins over "american".
        self.aliases = sorted(aliases.items(), key=lambda kv: -len(kv[0]))
        self._quarter_of = {}
        for doc in metas:
            m = _DOC_QUARTER.search(doc + "_")
            if m:
                self._quarter_of[doc] = (int(m.group(1)), int(m.group(2)))

    def companies_in(self, text: str) -> set[str]:
        padded = f" {_norm(text)} "
        found = set()
        for alias, company in self.aliases:
            if f" {alias} " in padded:
                found.add(company)
                padded = padded.replace(f" {alias} ", " ")
        return found

    def company_terms(self, companies: set[str]) -> set[str]:
        """Every word of every name or alias of these companies."""
        return {w for alias, c in self.aliases if c in companies for w in alias.split()}

    @staticmethod
    def years_in(text: str) -> set[int]:
        low = text.lower()
        years = {int(y) for y in _YEAR.findall(low)}
        years |= {2000 + int(y) for y in _SHORT_FY.findall(low)}
        return years

    @staticmethod
    def quarters_in(text: str) -> set[tuple[int, int]]:
        found = set()
        for i, rx in enumerate(_QUARTER):
            for m in rx.finditer(text):
                a, b = m.group(1), m.group(2)
                if i == 0:
                    q, y = int(a), int(b)
                elif i == 1:
                    y, q = int(a), int(b)
                else:
                    q, y = _ORDINAL[a.lower()], int(b)
                found.add((y + 2000 if y < 100 else y, q))
        return found

    def allowed_docs(self, text: str, level: str, quarters: bool = True) -> set[str] | None:
        if level not in LEVELS:
            raise ValueError(f"unknown filter level {level!r}; expected one of {LEVELS}")
        if level == "none":
            return None
        companies = self.companies_in(text)
        if not companies:
            return None
        docs = {d for d, m in self.metas.items() if m.company in companies}
        if level == "company_year":
            years = self.years_in(text)
            narrowed = {d for d in docs if years & set(self.metas[d].periods)}
            if narrowed:
                docs = narrowed
            found = self.quarters_in(text) if quarters else set()
            if found:
                by_quarter = {d for d in docs if self._quarter_of.get(d) in found}
                if by_quarter:
                    docs = by_quarter
        return docs


def load_aliases(path: Path) -> dict[str, list[str]]:
    """Extra names per company, from a YAML file of ``Company name: [alias, ...]``."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf8")) or {}
    return {company: [str(a) for a in names] for company, names in raw.items()}
