"""Regression gate: the BM25 pipeline on a fixed set of real filings must keep its Hit@10.

The gate runs the full BM25 system (filter, quarter, statement routing, company name
dropped) on the 50 dev questions, over a corpus of just the 40 filings those questions are
about. That corpus downloads and extracts in about a minute, so CI can run it on every push.
Scores on 40 filings aren't the scores on 360, so the gate compares the run with its own
recorded run in configs/gate.json, never with the full results.

The build fails when Hit@10 falls more than ``max_drop`` questions (one, by default) below
the recorded count. The one question of slack is there so that a single near-tie broken
differently by another numpy build doesn't fail CI. Questions that flip either way are
listed, so a change that trades one question for another shows up even when it passes.

MiniLM isn't in the gate: embedding the 40 filings would add a long step to every push, and
the dense code is covered by the unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from .data import Question, load_questions
from .evaluate import DEPTH, git, git_dirty
from .filters import load_aliases
from .metrics import score_ranking
from .search import FULL, Searcher

CONFIG = "bm25/" + FULL.key()


def define(questions: list[Question], dev: list[str]) -> dict:
    by_id = {q.qid: q for q in questions}
    return {"questions": sorted(dev), "filings": sorted({by_id[i].doc_name for i in dev})}


def run(data_dir: Path, gate: dict, aliases: Path | None) -> dict[str, bool]:
    """Whether each gate question has a gold page in the top 10."""
    searcher = Searcher.build(Path(data_dir), load_aliases(aliases) if aliases else {})
    missing = sorted(set(gate["filings"]) - set(searcher.corpus.docs))
    if missing:
        raise SystemExit(f"gate filings not extracted: {', '.join(missing[:5])}")
    by_id = {q.qid: q for q in load_questions(Path(data_dir))}
    hits = {}
    for qid in gate["questions"]:
        q = by_id[qid]
        ranks = score_ranking(searcher.ranked_pages(q.text, DEPTH, "bm25", FULL), q)["gold_ranks"]
        hits[qid] = any(r <= 10 for r in ranks)
    return hits


def record(gate: dict, hits: dict[str, bool]) -> dict:
    return {**gate, "recorded": {"config": CONFIG, "git_sha": git("rev-parse", "HEAD"),
                                 "git_dirty": git_dirty(),
                                 "hit@10": sum(hits.values()) / len(hits),
                                 "hits": sorted(q for q, h in hits.items() if h)}}


def check(gate: dict, hits: dict[str, bool]) -> tuple[bool, str]:
    if "recorded" not in gate:
        raise SystemExit("no recorded run to compare with; run `finrag gate --record` first")
    rec = gate["recorded"]
    before = set(rec["hits"])
    now = {q for q, h in hits.items() if h}
    lost, gained = sorted(before - now), sorted(now - before)
    allowed = len(before) - gate["max_drop"]
    ok = len(now) >= allowed
    lines = [f"{CONFIG} on {len(hits)} dev questions over {len(gate['filings'])} filings: "
             f"Hit@10 {len(now)}/{len(hits)}, recorded {len(before)}/{len(hits)} at "
             f"{(rec['git_sha'] or 'unknown')[:7]}, minimum {allowed}"]
    lines += [f"  lost:   {q}" for q in lost] + [f"  gained: {q}" for q in gained]
    lines.append("PASS" if ok else f"FAIL: more than {gate['max_drop']} question(s) below the recorded run")
    return ok, "\n".join(lines)


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def save(gate: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(gate, indent=1) + "\n")
