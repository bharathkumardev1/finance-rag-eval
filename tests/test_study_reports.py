"""The generation and reranker sections of RESULTS.md, built from synthetic runs."""

import json
from pathlib import Path

import numpy as np
from test_generate import ALL, NAIVE, QUESTIONS, RETRIEVED, FakeAPI, page_text

from finrag.data import Evidence, Question
from finrag.generate import Budget, Recorder, agreement, run
from finrag.report import generation, labels_status, rerank
from finrag.rerank import study

META = {"command": "finrag generate", "git_sha": "a" * 40, "git_dirty": False, "date": "2026-10-28",
        "model": "BAAI/bge-reranker-base", "cpu": "test cpu"}


def _generation(tmp_path):
    res = run(QUESTIONS, RETRIEVED, page_text, Recorder(tmp_path / "c.jsonl", FakeAPI(), Budget(25)),
              naive=NAIVE, settings=ALL)
    return {"meta": META, "retrieval": "sys", "naive_retrieval": "bm25/all-pages", "split": "test",
            "limit": None, **res}


def test_the_generation_section_names_every_setting_and_says_who_made_the_labels(tmp_path):
    res = _generation(tmp_path)
    labels = [{"qid": "q1", "setting": "gold", "your_verdict": "correct"},
              {"qid": "q2", "setting": "naive", "your_verdict": "incorrect"}]
    res["judge_check"] = {"labels": "results/generation/labels.csv", **agreement(
        {"rows": [r | {"verdict": "correct" if r["qid"] == "q1" else "incorrect"} for r in res["rows"]]},
        labels, n_boot=200)}
    text = generation(res, "pending an independent grade.")
    for phrase in ("gold pages", "reported system's top 5", "BM25 over all pages, top 5", "no pages",
                   "Sign-test p", "Accuracy by whether a gold page", "How the labels were made: pending",
                   "labels in `results/generation/labels.csv`"):
        assert phrase in text, phrase
    assert "hand labels" not in text and "a person" not in text
    assert "How the labels were made" not in generation(res)  # no status file, no claim


def test_the_reranker_section_reports_every_comparison_with_its_split(tmp_path):
    qs = [Question(f"q{i}", f"capex {i}", "", "", "D", "t", (Evidence("D", 2),)) for i in range(4)]
    ev = {"questions": {q.qid: {"split": "dev" if i < 2 else "test", "type": "t", "runs": {
        "x": {"top": [["D", 0], ["D", 2]]}, "y": {"top": [["D", 2], ["D", 1]]}}}
        for i, q in enumerate(qs)}}
    res = study(ev, qs, lambda d, p: "capex" if p == 2 else "o",
                lambda q, ps: np.array([float(p == "capex") for p in ps]), ["x", "y"])
    res["meta"] = META
    res["seconds_per_pair"] = 0.2
    text = rerank(res)
    assert "| Split | A | B | Metric |" in text and "`x+rerank` | `y` | hit@10" in text
    assert "200 ms per (question, page) pair" in text and "reranking 30 pages adds about 6.0 s" in text
    json.dumps(res)  # it must serialize


def test_labels_status_reads_the_status_line(tmp_path):
    assert labels_status(tmp_path) is None
    (tmp_path / "generation").mkdir()
    (tmp_path / "generation" / "LABELS.md").write_text("# Labels\n\nStatus: graded blind by the author.\n")
    assert labels_status(tmp_path) == "graded blind by the author."


def test_partial_runs_must_not_overwrite_the_real_recording(capsys):
    from finrag.cli import main

    assert main(["generate", "--limit", "5"]) == 1
    assert "own --record and --out" in capsys.readouterr().err
    assert main(["generate", "--settings", "gold,telepathy"]) == 1
    assert main(["rerank-study", "--limit", "3"]) == 1
    assert "own --out" in capsys.readouterr().err


def test_small_p_values_and_the_cap_print_as_they_are(tmp_path):
    from finrag.generate import format_p

    assert format_p(7.6e-05) == "<0.001" and format_p(0.0574) == "0.057" and format_p(1.0) == "1.000"
    res = _generation(tmp_path)
    res["cap_usd"] = 1.5
    text = generation(res)
    assert "of a $1.50 cap" in text and "$2 cap" not in text


def _windowed(ranks_by_name):
    qs = {f"q{i}": {"split": "dev" if i < 2 else "test", "type": "t", "runs": {
        name: {"gold_ranks": ranks[i]} for name, ranks in ranks_by_name.items()}} for i in range(4)}
    return qs


def test_the_windows_section_compares_with_no_rerank_and_with_one_window():
    from finrag.report import rerank_windows

    base, name = "x", "x+rerank"
    one = {"questions": _windowed({base: [[1], [2], [1], []], name: [[5], [9], [3], []]})}
    many = {"base": base, "windows": 3, "pairs_scored": 40, "scoring_seconds": 20.0, "seconds_per_pair": 0.5,
            "questions": _windowed({base: [[1], [2], [1], []], name: [[4], [12], [3], []]})}
    text = rerank_windows(one, many)
    assert "### Reading 3 windows of each page" in text and "500 ms per page" in text
    assert "| dev | 3 windows | no rerank | hit@10 | -0.500 |" in text
    assert "| test | 3 windows | 1 window | mrr@10 | +0.000 |" in text
    assert text.count("| 1 window |") >= 2


def test_the_report_adds_the_windows_section_only_when_that_file_exists(tmp_path):
    import json

    from finrag.report import write_markdown

    results = json.loads((Path(__file__).resolve().parents[1] / "results" / "eval.json").read_text())
    one = {"meta": META, "bases": ["x"], "base": "x", "top": 30, "windows": 1, "table": [], "comparisons": [],
           "questions": _windowed({"x": [[1]] * 4, "x+rerank": [[2]] * 4}), "seconds_per_pair": None}
    many = one | {"windows": 3, "pairs_scored": 10, "scoring_seconds": 5.0, "seconds_per_pair": 0.5}
    (tmp_path / "rerank.json").write_text(json.dumps(one))
    out = tmp_path / "RESULTS.md"
    write_markdown(results, out, tmp_path)
    assert "### Reading" not in out.read_text()
    (tmp_path / "rerank-windows3.json").write_text(json.dumps(many))
    write_markdown(results, out, tmp_path)
    assert out.read_text().count("### Reading 3 windows") == 1
