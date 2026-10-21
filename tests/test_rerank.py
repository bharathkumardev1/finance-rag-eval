import numpy as np
import pytest

from finrag.data import Evidence, Question
from finrag.rerank import rerank, study


def test_rerank_orders_the_head_by_score_and_keeps_the_tail():
    ranked = [["A", 0], ["A", 1], ["A", 2], ["A", 3]]
    assert rerank(ranked, np.array([0.1, 0.9, 0.1])) == [["A", 1], ["A", 0], ["A", 2], ["A", 3]]


def test_study_scores_the_base_and_the_reranked_ranking_alike():
    qs = [Question(f"q{i}", f"capex {i}", "", "", "D", "t", (Evidence("D", 2),)) for i in range(4)]
    evaluation = {"questions": {
        q.qid: {"split": "dev" if i < 2 else "test", "type": "t",
                "runs": {"base": {"top": [["D", 0], ["D", 1], ["D", 2]]}}}
        for i, q in enumerate(qs)}}

    def page_text(doc, page):
        return "capital expenditure capex" if page == 2 else "other"

    def score(question, passages):
        return np.array([sum(w in p for w in question.split()) for p in passages], dtype=float)

    res = study(evaluation, qs, page_text, score, "base")
    rows = {(r["config"], r["split"]): r for r in res["table"]}
    assert rows["base", "test"]["hit@1"] == 0 and rows["base+rerank", "test"]["hit@1"] == 1
    hit1_test = [c for c in res["comparisons"] if c["split"] == "test" and c["metric"] == "hit@1"][0]
    assert hit1_test["diff"] == pytest.approx(1.0) and hit1_test["a_better"] == 2
