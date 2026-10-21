import json

import pytest

from finrag.cli import main
from finrag.data import load_questions
from finrag.evaluate import compare, run
from finrag.metrics import gold_pages
from finrag.report import write_markdown
from finrag.split import load_split


@pytest.fixture
def questions(mini_dir):
    return load_questions(mini_dir)


def test_run_scores_each_system_on_each_split(mini_dir, questions):
    split = load_split(mini_dir / "split.json")
    gold = {q.text: sorted(gold_pages(q)) for q in questions}
    systems = {"oracle": lambda text: gold[text], "empty": lambda text: []}
    res = run(systems, questions, split)
    rows = {(r["config"], r["split"]): r for r in res["table"]}
    assert len(rows) == 6
    assert rows["oracle", "test"]["hit@10"] == 1.0 and rows["oracle", "test"]["mrr@10"] == 1.0
    assert rows["empty", "all"]["hit@10"] == 0.0 and rows["empty", "all"]["n"] == len(questions)
    assert rows["oracle", "dev"]["n"] == len(split["dev"])
    assert rows["oracle", "test"]["p95_ms"] >= rows["oracle", "test"]["p50_ms"] >= 0

    d = compare(res, "oracle", "empty")
    assert d["diff"] == 1.0 and d["a_better"] == len(split["test"]) and d["b_better"] == 0


def test_systems_only_see_the_question_text(mini_dir, questions):
    seen = []
    run({"spy": lambda text: seen.append(text) or []}, questions, load_split(mini_dir / "split.json"))
    assert seen == [q.text for q in questions]


def test_eval_command_writes_results_and_markdown(mini_dir, tmp_path):
    out, md = tmp_path / "eval.json", tmp_path / "RESULTS.md"
    code = main(["eval", "--data-dir", str(mini_dir), "--split", str(mini_dir / "split.json"),
                 "--out", str(out), "--markdown", str(md)])
    assert code == 0
    res = json.loads(out.read_text())
    meta = res["meta"]
    assert meta["command"].startswith("finrag eval --data-dir")
    assert len(meta["git_sha"] or "x" * 40) == 40
    assert (meta["filings"], meta["pages"], meta["questions"]) == (7, 6 * 8 + 3, 12)
    text = md.read_text()
    assert "## Test (9 questions)" in text and "| `bm25/all-pages` |" in text


def test_markdown_lists_comparisons_when_present(mini_dir, questions, tmp_path):
    split = load_split(mini_dir / "split.json")
    gold = {q.text: sorted(gold_pages(q)) for q in questions}
    res = run({"a": lambda t: gold[t], "b": lambda t: []}, questions, split)
    res["meta"] = {"git_sha": "0" * 40, "git_dirty": False, "command": "finrag eval",
                   "date": "2026-01-01", "financebench_commit": "f" * 40, "filings": 7, "pages": 51,
                   "questions": 12, "dev": 3, "test": 9, "cpu": "cpu", "cpus": 1,
                   "python": "3.13", "numpy": "2"}
    res["comparisons"] = [compare(res, "a", "b")]
    write_markdown(res, tmp_path / "R.md")
    text = (tmp_path / "R.md").read_text()
    assert "| `a` | `b` | hit@10 | +1.000 |" in text and "commit `0000000`" in text
