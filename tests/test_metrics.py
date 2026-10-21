import numpy as np
import pytest

from finrag.data import Evidence, Question
from finrag.metrics import bootstrap_ci, metrics, paired_diff, per_question, score_ranking


def question(*pages):
    return Question("x", "q", "", "", pages[0][0], "t", tuple(Evidence(d, p) for d, p in pages))


def test_score_ranking_finds_gold_pages_and_the_right_filing():
    q = question(("A_2022_10K", 4), ("A_2022_10K", 9))
    ranked = [("B_2022_10K", 1), ("A_2022_10K", 2), ("A_2022_10K", 9), ("A_2022_10K", 4)]
    assert score_ranking(ranked, q) == {"gold_ranks": [3, 4], "n_gold": 2, "doc_rank": 2}
    assert score_ranking([("B_2022_10K", 1)], q)["doc_rank"] is None


def test_metrics_from_ranks():
    recs = [{"gold_ranks": [1], "n_gold": 1, "doc_rank": 1},
            {"gold_ranks": [4, 12], "n_gold": 2, "doc_rank": 2},
            {"gold_ranks": [], "n_gold": 1, "doc_rank": None}]
    m = metrics(recs)
    assert m["hit@1"] == pytest.approx(1 / 3) and m["hit@5"] == pytest.approx(2 / 3)
    assert m["recall@10"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert m["mrr@10"] == pytest.approx((1 + 0.25) / 3)
    assert m["doc_hit@1"] == pytest.approx(1 / 3)
    assert m["n_multi_page"] == 1 and m["recall@10_multi_page"] == pytest.approx(0.5)
    np.testing.assert_allclose(per_question(recs, "hit@5"), [1, 1, 0])


def test_bootstrap_interval_covers_the_mean():
    vals = np.array([1.0] * 40 + [0.0] * 60)
    lo, hi = bootstrap_ci(vals)
    assert lo < 0.4 < hi and hi - lo < 0.25


def test_paired_difference_counts_wins_each_way():
    a = np.array([1.0, 1.0, 0.0, 1.0])
    b = np.array([0.0, 1.0, 1.0, 0.0])
    d = paired_diff(a, b)
    assert d["diff"] == pytest.approx(0.25) and (d["a_better"], d["b_better"]) == (2, 1)
