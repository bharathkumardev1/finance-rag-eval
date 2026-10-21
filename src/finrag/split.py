"""The one dev/test split every result in this repo uses.

Anything chosen by looking at results (the alias table, which retriever ships) is chosen on
dev. Headline numbers are on test. The split is stratified by FinanceBench's question type
and ordered by sha256 of the question id, so it is fixed without a random seed.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .data import Question

DEV_PER_TYPE = {"domain-relevant": 17, "metrics-generated": 17, "novel-generated": 16}


def make_split(questions: list[Question], dev_per_type: dict[str, int] | None = None
               ) -> dict[str, list[str]]:
    dev_per_type = dev_per_type or DEV_PER_TYPE
    by_type: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        by_type[q.question_type].append(q)
    dev: set[str] = set()
    for qtype, group in by_type.items():
        group = sorted(group, key=lambda q: hashlib.sha256(q.qid.encode()).hexdigest())
        dev.update(q.qid for q in group[: dev_per_type[qtype]])
    return {"dev": sorted(dev), "test": sorted(q.qid for q in questions if q.qid not in dev)}


def load_split(path: Path) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text())


def save_split(split: dict[str, list[str]], path: Path) -> None:
    Path(path).write_text(json.dumps(split, indent=1) + "\n")
