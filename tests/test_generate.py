import json

import pytest

from finrag.data import Evidence, Question
from finrag.generate import (
    DECLINE,
    Budget,
    BudgetExceeded,
    IncompleteReply,
    MissingRecording,
    Recorder,
    Reply,
    agreement,
    cost,
    label_sheet,
    price,
    run,
)

QUESTIONS = [
    Question("q1", "What was Acme's FY2021 revenue?", "$2.4 billion", "Acme", "ACME_2021_10K", "t1",
             (Evidence("ACME_2021_10K", 5),)),
    Question("q2", "Who audits Acme?", "Ernst & Young", "Acme", "ACME_2021_10K", "t2",
             (Evidence("ACME_2021_10K", 9),)),
]
RETRIEVED = {"q1": [["ACME_2021_10K", 5], ["ACME_2021_10K", 1]], "q2": [["ACME_2021_10K", 1]]}


def page_text(doc, page):
    return f"{doc} page {page}: revenue $2,400 million" if page == 5 else f"{doc} page {page}: other"


class FakeAPI:
    """Answers revenue questions, declines the rest; the judge marks '$2.4 billion' correct."""

    def __init__(self):
        self.calls = []

    def __call__(self, request):
        self.calls.append(request)
        if "text" in request:  # the judge
            ok = "2.4 billion" in request["input"].split("Answer to grade:")[1]
            text = json.dumps({"verdict": "correct" if ok else "incorrect", "reason": "r"})
            model = "gpt-6.1-sol"
        else:
            text = "$2.4 billion (page 6)" if "revenue $2,400" in request["input"] else DECLINE
            model = "gpt-6-luna-2026-05-18"
        return Reply(text, model, 1000, 200, 300, 100, 0.5, "completed")


def test_cost_uses_cached_and_output_prices():
    r = Reply("x", "gpt-6-luna", 1_000_000, 400_000, 100_000, 0, 1.0, "completed")
    assert cost(r) == pytest.approx(0.6 * 0.10 + 0.4 * 0.01 + 0.1 * 0.50)
    assert price("gpt-6-luna-2026-05-18") == price("gpt-6-luna")
    with pytest.raises(KeyError):
        price("some-other-model")


def test_run_scores_both_settings_and_records_every_call(tmp_path):
    api = FakeAPI()
    rec = Recorder(tmp_path / "calls.jsonl", api, Budget(25))
    res = run(QUESTIONS, RETRIEVED, page_text, rec)
    verdicts = {(r["qid"], r["setting"]): r["verdict"] for r in res["rows"]}
    assert verdicts == {("q1", "retrieved"): "correct", ("q1", "gold"): "correct",
                        ("q2", "retrieved"): "declined", ("q2", "gold"): "declined"}
    assert res["summary"]["settings"]["gold"]["accuracy"] == 0.5
    # q1's two answers are the same text, so the second judge call is the first one replayed
    assert res["calls"] == {"made": 5, "replayed": 1} and len(api.calls) == 5
    assert res["rows"][0]["answer_model"] == "gpt-6-luna-2026-05-18"

    replay = Recorder(tmp_path / "calls.jsonl", None, Budget(25))
    again = run(QUESTIONS, RETRIEVED, page_text, replay)
    assert again["rows"] == res["rows"] and again["calls"]["made"] == 0


def test_replay_without_a_recording_fails_loudly(tmp_path):
    with pytest.raises(MissingRecording):
        run(QUESTIONS, RETRIEVED, page_text, Recorder(tmp_path / "none.jsonl", None, Budget(25)))


def test_the_cap_stops_the_run_before_it_is_passed(tmp_path):
    api = FakeAPI()
    res = run(QUESTIONS, RETRIEVED, page_text, Recorder(tmp_path / "c.jsonl", api, Budget(0.0025)))
    assert res["stopped"] and res["recorded_spend_usd"] <= 0.0025
    spent = res["recorded_spend_usd"]
    again = Recorder(tmp_path / "c.jsonl", api, Budget(0.0025))
    assert again.budget.spent == pytest.approx(spent, abs=1e-6)  # the cap counts what's recorded


def test_label_sheet_hides_the_verdict_and_agreement_counts_both_ways(tmp_path):
    res = run(QUESTIONS, RETRIEVED, page_text, Recorder(tmp_path / "c.jsonl", FakeAPI(), Budget(25)))
    sheet = label_sheet(res, {q.qid: q for q in QUESTIONS}, n=10)
    assert len(sheet) == 2 and all("verdict" not in k for k in sheet[0] if k != "your_verdict")
    sheet[0]["your_verdict"] = "correct"
    sheet[1]["your_verdict"] = "incorrect"
    check = agreement(res, sheet)
    assert check["labeled"] == 2 and check["agree"] == 1 and check["judge_says_correct_you_dont"] == 1
    with pytest.raises(BudgetExceeded):
        Budget(0.0).check({"model": "gpt-6-luna", "instructions": "x", "input": "y",
                           "max_output_tokens": 10})


def test_a_cut_off_reply_stops_the_run_counts_toward_the_cap_and_is_asked_again(tmp_path):
    class CutsOffTheFirstJudge(FakeAPI):
        def __call__(self, request):
            reply = super().__call__(request)
            if "text" in request and sum("text" in r for r in self.calls) == 1:
                reply.status, reply.text = "incomplete", '{"verdict": "corr'
            return reply

    path = tmp_path / "calls.jsonl"
    res = run(QUESTIONS, RETRIEVED, page_text, Recorder(path, CutsOffTheFirstJudge(), Budget(25)))
    assert res["stopped"] and "incomplete" in res["stopped"]
    assert any(json.loads(line).get("incomplete") for line in path.read_text().splitlines())
    first = Recorder(path, None, Budget(25)).budget.spent
    assert first > 0
    api = FakeAPI()
    res = run(QUESTIONS, RETRIEVED, page_text, Recorder(path, api, Budget(25)))
    assert not res["stopped"] and len(res["rows"]) == 4
    assert sum("text" in r for r in api.calls) == 1  # only the cut-off judge call is made again
    judge = {"model": "gpt-6.1-sol", "instructions": "", "input": "Answer to grade: x",
             "max_output_tokens": 10, "text": {}}
    with pytest.raises(IncompleteReply):
        Recorder(tmp_path / "x.jsonl", CutsOffTheFirstJudge(), Budget(25)).call(judge, {"qid": "q"})


def test_committed_summary_recomputes_from_its_rows():
    from pathlib import Path

    from finrag.generate import summarize

    path = Path(__file__).resolve().parents[1] / "results" / "generation.json"
    if not path.exists():
        pytest.skip("no generation results committed yet")
    res = json.loads(path.read_text())
    assert summarize(res["rows"]) == res["summary"]
