import numpy as np
import pytest

from finrag.bm25 import BM25, STOPWORDS, tokenize

DOCS = [
    "Capital expenditures were $120 million in fiscal 2021, up from 2020.",
    "Wildfire mitigation spending totaled $705 million; wildfire risk remains.",
    "Subscription revenue grew 18 percent to $1,040 million.",
    "",
    "The company repurchased shares under its buyback program and the program continues.",
]


def test_tokenize_matches_lucene_style_rules():
    assert tokenize("The FY2021 capex, in $ millions: 1,234!") == ["fy2021", "capex", "millions",
                                                                   "234"]
    assert "the" in STOPWORDS and len(STOPWORDS) == 33
    assert tokenize("a I x") == []  # single characters never become tokens


def test_scores_match_bm25s():
    bm25s = pytest.importorskip("bm25s")
    ref = bm25s.BM25()
    ref.index(bm25s.tokenize(DOCS, stopwords="en", show_progress=False), show_progress=False)
    ours = BM25(tokenize(d) for d in DOCS)
    for query in ("wildfire mitigation wildfire", "capital expenditures 2021",
                  "program buyback shares", "revenue"):
        q = tokenize(query)
        expected = np.asarray(ref.get_scores([t for t in q if t in ref.vocab_dict]))
        np.testing.assert_allclose(ours.scores(q), expected, atol=1e-5)


def test_scoring_a_subset_equals_scoring_everything(tmp_path):
    index = BM25(tokenize(d) for d in DOCS)
    rows = np.array([0, 2, 4])
    q = tokenize("program revenue capital")
    np.testing.assert_allclose(index.scores(q, rows), index.scores(q)[rows])
    assert index.scores(tokenize("unknownword")).sum() == 0
    index.save(tmp_path / "idx")
    loaded = BM25.load(tmp_path / "idx")
    np.testing.assert_allclose(loaded.scores(q), index.scores(q))
    assert loaded.avg_len == pytest.approx(index.avg_len)
