import pytest

from finrag import gate as g
from finrag.data import load_questions
from finrag.split import load_split

GATE = {"max_drop": 1, "questions": ["a", "b", "c", "d"], "filings": ["X"],
        "recorded": {"git_sha": "1234567" + "0" * 33, "hits": ["a", "b", "c"]}}


def test_one_question_of_slack():
    ok, report = g.check(GATE, {"a": True, "b": True, "c": False, "d": False})
    assert ok and "lost:   c" in report and report.endswith("PASS")
    ok, report = g.check(GATE, {"a": True, "b": False, "c": False, "d": True})
    assert ok and "gained: d" in report
    ok, report = g.check(GATE, {"a": True, "b": False, "c": False, "d": False})
    assert not ok and "FAIL" in report


def test_a_gate_without_a_recording_says_what_to_do():
    with pytest.raises(SystemExit, match="finrag gate --record"):
        g.check({k: v for k, v in GATE.items() if k != "recorded"}, {})


def test_define_record_and_check_on_the_fixture(mini_dir, tmp_path):
    dev = load_split(mini_dir / "split.json")["dev"]
    gate = {"max_drop": 1, **g.define(load_questions(mini_dir), dev)}
    assert gate["questions"] == sorted(dev) and len(gate["filings"]) <= len(dev)
    hits = g.run(mini_dir, gate, None)
    recorded = g.record(gate, hits)
    assert recorded["recorded"]["hit@10"] == sum(hits.values()) / len(dev)
    g.save(recorded, tmp_path / "gate.json")
    ok, _ = g.check(g.load(tmp_path / "gate.json"), hits)
    assert ok
