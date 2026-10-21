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
FILINGS = {
    "A": [tokenize(DOCS[0]), tokenize(DOCS[1])],
    "B": [tokenize(DOCS[2])],
    "C": [tokenize(DOCS[3]), tokenize(DOCS[4])],
}
OFFSETS = {"A": (0, 2), "B": (2, 3), "C": (3, 5)}
QUERIES = ["wildfire mitigation wildfire", "capital expenditures 2021", "program buyback shares",
           "revenue", "million"]


def test_tokenize_matches_lucene_style_rules():
    assert tokenize("The FY2021 capex, in $ millions: 1,234!") == ["fy2021", "capex", "millions",
                                                                   "234"]
    assert "the" in STOPWORDS and len(STOPWORDS) == 33
    assert tokenize("a I x") == []  # single characters never become tokens


def test_scores_match_bm25s():
    bm25s = pytest.importorskip("bm25s")
    ref = bm25s.BM25()
    ref.index(bm25s.tokenize(DOCS, stopwords="en", show_progress=False), show_progress=False)
    ours = BM25.from_pages(tokenize(d) for d in DOCS)
    for query in QUERIES:
        q = tokenize(query)
        expected = np.asarray(ref.get_scores([t for t in q if t in ref.vocab_dict]))
        np.testing.assert_allclose(ours.scores(q), expected, atol=1e-5)


def test_scoring_a_subset_equals_scoring_everything():
    index = BM25.from_pages(tokenize(d) for d in DOCS)
    rows = np.array([0, 2, 4])
    q = tokenize("program revenue capital")
    np.testing.assert_array_equal(index.scores(q, rows), index.scores(q)[rows])
    assert index.scores(tokenize("unknownword")).sum() == 0


def test_segments_score_exactly_like_one_pass(tmp_path):
    one = BM25()
    one.add(FILINGS)
    one.assign_rows(OFFSETS)
    grown = BM25()
    grown.add({"C": FILINGS["C"]})
    grown.save(tmp_path)
    grown = BM25.load(tmp_path)
    grown.add({"A": FILINGS["A"]})
    grown.add({"B": FILINGS["B"]})
    grown.save(tmp_path)
    grown = BM25.load(tmp_path)
    grown.assign_rows(OFFSETS)
    assert len(grown.segments) == 3 and grown.n_docs == one.n_docs == 5
    for query in QUERIES:
        q = tokenize(query)
        np.testing.assert_array_equal(grown.scores(q), one.scores(q))


def test_save_writes_only_new_segments(tmp_path):
    index = BM25()
    index.add({"A": FILINGS["A"]})
    index.save(tmp_path)
    first = (tmp_path / "segment-0000.npz").stat().st_mtime_ns
    index.add({"B": FILINGS["B"]})
    index.save(tmp_path)
    assert (tmp_path / "segment-0000.npz").stat().st_mtime_ns == first
    assert sorted(p.name for p in tmp_path.glob("segment-*.npz")) == ["segment-0000.npz",
                                                                       "segment-0001.npz"]


def test_a_filing_goes_in_once_and_must_match_the_corpus():
    index = BM25()
    index.add({"A": FILINGS["A"]})
    with pytest.raises(ValueError, match="already indexed: A"):
        index.add({"A": FILINGS["A"]})
    with pytest.raises(ValueError, match="indexed 2 pages, corpus has 3"):
        index.assign_rows({"A": (0, 3)})
