import json

import numpy as np
import pytest

from finrag import evaluate as ev
from finrag.data import Evidence, Question, load_questions
from finrag.dense import DenseIndex
from finrag.failures import classify
from finrag.search import Options, rrf, top_k


def q(text, doc, page, qtype="t", qid="x"):
    return Question(qid, text, "", "", doc, qtype, (Evidence(doc, page),))


def test_top_k_breaks_ties_by_position():
    assert top_k(np.array([1.0, 3.0, 3.0, 2.0]), 3).tolist() == [1, 2, 3]
    assert top_k(np.array([]), 3).tolist() == []


def test_rrf_rewards_agreement():
    rows, scores = rrf([np.array([10, 11, 12]), np.array([12, 10, 13])])
    assert rows[0] == 10 and set(rows.tolist()) == {10, 11, 12, 13}
    assert scores[0] == pytest.approx(1 / 61 + 1 / 62)


def test_filter_and_routing_on_the_fixture(searcher):
    question = "What is the quick ratio of Globex for FY2021?"
    hits, info = searcher.search(question, "bm25", 3)
    assert info["filings"] == ["GLOBEX_2021_10K"] and info["statements"] == ["balance"]
    assert (hits[0].doc, hits[0].page) == ("GLOBEX_2021_10K", 4)
    hits, _ = searcher.search(question, "bm25", 3, Options(route_statements=False))
    assert (hits[0].doc, hits[0].page) != ("GLOBEX_2021_10K", 4)  # the table shares no words
    hits, info = searcher.search("What was Acme's net revenue by segment in 2022 Q2?", "bm25", 1)
    assert info["filings"] == ["ACME_2022Q2_10Q"] and hits[0].page == 1


def test_dropping_company_terms_changes_the_bm25_query(searcher):
    full = searcher.query_tokens("How many shares did Acme repurchase?", True, Options())
    kept = searcher.query_tokens("How many shares did Acme repurchase?", True,
                                 Options(drop_company_terms=False))
    assert "acme" in kept and "acme" not in full


def test_dense_maxp_scores_subsets_consistently(dense_searcher):
    d = dense_searcher.dense
    qv = dense_searcher.encoder.encode(["wildfire mitigation spending"])[0]
    full = d.scores(qv)
    rows = np.array([1, 3, 4, 20, 21])
    np.testing.assert_allclose(d.scores(qv, rows), full[rows], rtol=1e-6)
    p = 8  # a page's score is its best window
    lo, hi = d.page_start[p], d.page_end[p]
    assert full[p] == pytest.approx(float((d.vectors[lo:hi] @ qv).max()))


def test_dense_index_rejects_pages_without_vectors():
    with pytest.raises(ValueError):
        DenseIndex(np.ones((2, 4), dtype=np.float32), np.array([0, 0]), n_pages=2)


def test_hybrid_and_dense_rank_every_filtered_page(dense_searcher):
    question = "What was Globex's FY2022 capital expenditure according to the cash flow statement?"
    for method in ("dense", "hybrid"):
        hits, info = dense_searcher.search(question, method, 50)
        assert len(hits) == info["pages_searched"] == 8
        assert (hits[0].doc, hits[0].page) == ("GLOBEX_2022_10K", 6)


def test_metrics_from_ranks():
    recs = [{"gold_ranks": [1], "n_gold": 1, "doc_rank": 1},
            {"gold_ranks": [4, 12], "n_gold": 2, "doc_rank": 2},
            {"gold_ranks": [], "n_gold": 1, "doc_rank": None}]
    m = ev.metrics(recs)
    assert m["hit@1"] == pytest.approx(1 / 3) and m["hit@5"] == pytest.approx(2 / 3)
    assert m["recall@10"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert m["mrr@10"] == pytest.approx((1 + 0.25) / 3)
    assert m["doc_hit@1"] == pytest.approx(1 / 3)
    np.testing.assert_allclose(ev.per_question(recs, "hit@5"), [1, 1, 0])


def test_bootstrap_intervals_cover_the_mean():
    vals = np.array([1.0] * 40 + [0.0] * 60)
    lo, hi = ev.bootstrap_ci(vals)
    assert lo < 0.4 < hi and hi - lo < 0.25
    d = ev.paired_diff(vals, np.zeros(100))
    assert d["diff"] == pytest.approx(0.4) and d["a_better"] == 40 and d["b_better"] == 0


def test_split_is_stratified_and_deterministic(mini_dir):
    qs = load_questions(mini_dir)
    per = {"domain-relevant": 1, "metrics-generated": 1, "novel-generated": 1}
    a, b = ev.make_split(qs, per), ev.make_split(list(reversed(qs)), per)
    assert a == b and len(a["dev"]) == 3 and len(a["test"]) == 9
    assert a == json.loads((mini_dir / "split.json").read_text())


def test_failure_categories(searcher):
    opts = Options()
    nothing = q("Which segment grew fastest last year?", "ACME_2021_10K", 1)
    assert classify(searcher, nothing, [], opts) == "no_company"
    wrong_year = q("What did Acme do in 2022?", "ACME_2021_10K", 1)
    assert classify(searcher, wrong_year, [], opts) == "filter_dropped_filing"
    unflagged = q("How did Acme do in FY2021?", "ACME_2021_10K", 4)
    assert classify(searcher, unflagged, [], opts) == "statement_not_detected"
