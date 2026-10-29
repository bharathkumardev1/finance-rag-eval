import json

import pytest

from finrag.data import Evidence, Question
from finrag.generate import (
    CLOSED_BOOK_INSTRUCTIONS,
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
    sign_test_p,
    summarize,
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


def test_label_sheet_hides_the_verdict_and_lists_an_answer_given_in_the_same_words_once(tmp_path):
    res = run(QUESTIONS, RETRIEVED, page_text, Recorder(tmp_path / "c.jsonl", FakeAPI(), Budget(25)))
    sheet = label_sheet(res, {q.qid: q for q in QUESTIONS}, n=10)
    # q1's retrieved and gold answers are the same words and q2 was declined, so one row
    assert len(sheet) == 1 and sheet[0]["qid"] == "q1"
    assert all("verdict" not in k for k in sheet[0] if k != "your_verdict")


def _rows(spec):
    return [{"qid": q, "setting": "gold", "verdict": v, "answer": a} for q, v, a in spec]


def test_agreement_counts_both_ways_and_resamples_questions():
    rows = _rows([("a", "correct", "x"), ("b", "incorrect", "x"), ("c", "correct", "x"),
                  ("d", "correct", "x"), ("e", "incorrect", "x"), ("f", "incorrect", "x")])
    labels = [{"qid": q, "setting": "gold", "your_verdict": v}
              for q, v in [("a", "correct"), ("b", "incorrect"), ("c", "incorrect"), ("d", "correct"),
                           ("e", "incorrect"), ("f", "correct")]]
    check = agreement({"rows": rows}, labels, n_boot=2000)
    assert check["labeled"] == 6 and check["questions"] == 6 and check["agree"] == 4
    assert check["judge_says_correct_you_dont"] == 1 and check["judge_says_incorrect_you_dont"] == 1
    assert check["agreement"] == pytest.approx(4 / 6)
    assert check["cohens_kappa"] == pytest.approx(1 / 3)  # po 4/6, chance 1/2
    lo, hi = check["agreement_ci95"]
    assert 0 <= lo <= check["agreement"] <= hi <= 1
    assert agreement({"rows": rows}, [{"qid": "a", "setting": "gold", "your_verdict": " "}]) == {"labeled": 0}
    with pytest.raises(BudgetExceeded):
        Budget(0.0).check({"model": "gpt-6-luna", "instructions": "x", "input": "y",
                           "max_output_tokens": 10})


def test_sign_test_is_exact_and_two_sided():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 1) == pytest.approx(14 / 64)
    assert sign_test_p(1, 5) == sign_test_p(5, 1)
    assert sign_test_p(10, 2) == pytest.approx(2 * 79 / 4096)
    assert sign_test_p(3, 3) == 1.0


NAIVE = {"q1": [["ACME_2021_10K", 1]], "q2": [["ACME_2021_10K", 9], ["ACME_2021_10K", 1]]}
ALL = ("gold", "retrieved", "naive", "closed_book")


def test_four_settings_use_their_own_pages_and_a_closed_book_prompt_has_none(tmp_path):
    api = FakeAPI()
    res = run(QUESTIONS, RETRIEVED, page_text, Recorder(tmp_path / "c.jsonl", api, Budget(25)),
              naive=NAIVE, settings=ALL)
    got = {(r["qid"], r["setting"]): r for r in res["rows"]}
    assert len(res["rows"]) == 8 and res["settings"] == list(ALL)
    assert got["q1", "closed_book"]["pages"] == [] and got["q1", "closed_book"]["verdict"] == "declined"
    assert got["q1", "naive"]["verdict"] == "declined"  # page 1 has no revenue
    assert got["q2", "naive"]["gold_in_pages"] is True and got["q2", "retrieved"]["gold_in_pages"] is False
    assert got["q1", "closed_book"]["gold_in_pages"] is False and got["q1", "gold"]["gold_in_pages"] is True
    closed = [r for r in api.calls if r["instructions"] == CLOSED_BOOK_INSTRUCTIONS]
    assert len(closed) == 2 and all("Filing pages" not in r["input"] for r in closed)
    s = res["summary"]
    assert set(s["settings"]) == set(ALL)
    assert [(c["a"], c["b"]) for c in s["comparisons"]] == [
        ("gold", "retrieved"), ("retrieved", "naive"), ("retrieved", "closed_book"), ("gold", "closed_book")]
    assert s["gold_minus_retrieved"]["diff"] == s["comparisons"][0]["diff"]
    assert s["settings"]["retrieved"]["gold_in_pages"] == {"n": 1, "accuracy": 1.0}
    assert s["settings"]["retrieved"]["gold_not_in_pages"] == {"n": 1, "accuracy": 0.0}
    assert summarize(res["rows"]) == s


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


def test_cost_totals_do_not_depend_on_how_the_python_version_adds_floats():
    rows = [{"qid": f"q{i}", "setting": "gold", "verdict": "correct", "cost_usd": c, "answer_seconds": 1.0,
             "type": "t", "gold_in_pages": True} for i, c in enumerate([0.1, 0.2, 0.3] * 4)]
    assert summarize(rows)["settings"]["gold"]["cost_usd"] == 2.4
