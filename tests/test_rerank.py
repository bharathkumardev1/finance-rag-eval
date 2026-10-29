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


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {w: i for i, w in enumerate(["[UNK]", *"abcdefghij"])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


def test_windows_cut_at_token_edges_and_drop_what_is_past_the_last():
    from finrag.rerank import split_windows

    text = "a b c d e f g"
    offsets = _tokenizer().encode(text, add_special_tokens=False).offsets
    assert split_windows(text, offsets, 3, 2) == ["a b c", "d e f"]  # g is past the second window
    assert split_windows(text, offsets, 3, 3) == ["a b c", "d e f", "g"]
    assert split_windows(text, offsets, 7, 3) == [text]  # fits: whole
    short = _tokenizer().encode("a b", add_special_tokens=False).offsets
    assert split_windows("a b", short, 1, 5) == ["a", "b"]
    with pytest.raises(ValueError):
        split_windows(text, offsets, 0, 2)


def test_a_page_scores_as_its_best_window():
    from finrag.rerank import best_per_owner

    out = best_per_owner(np.array([0.1, 0.9, 0.4, 0.2, 0.3]), [0, 0, 1, 2, 2], 4)
    assert out[:3].tolist() == [0.9, 0.4, 0.3] and out[3] == -np.inf


def test_systems_share_the_scores_of_pages_they_have_in_common():
    qs = [Question(f"q{i}", f"capex {i}", "", "", "D", "t", (Evidence("D", 2),)) for i in range(2)]
    evaluation = {"questions": {q.qid: {"split": "dev" if i == 0 else "test", "type": "t", "runs": {
        "x": {"top": [["D", 0], ["D", 1], ["D", 2]]},
        "y": {"top": [["D", 1], ["D", 2], ["D", 3]]}}} for i, q in enumerate(qs)}}
    seen = []

    def score(question, passages):
        seen.append(len(passages))
        return np.array([1.0 if p == "capex" else 0.0 for p in passages])

    res = study(evaluation, qs, lambda d, p: "capex" if p == 2 else "other", score, ["x", "y"], windows=2)
    assert seen == [3, 1, 3, 1]  # per question: x's three pages, then only y's page 3
    assert res["pairs_scored"] == 8 and res["windows"] == 2 and res["bases"] == ["x", "y"]
    names = {r["config"] for r in res["table"]}
    assert names == {"x", "x+rerank", "y", "y+rerank"}
    cross = [c for c in res["comparisons"] if c["a"] == "x+rerank" and c["b"] == "y"]
    assert {(c["split"], c["metric"]) for c in cross} == {
        ("dev", "hit@10"), ("dev", "mrr@10"), ("test", "hit@10"), ("test", "mrr@10")}
    assert res["questions"]["q0"]["top_after"]["x"][0] == ["D", 2]
