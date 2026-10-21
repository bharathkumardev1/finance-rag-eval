import numpy as np

from finrag.search import Corpus, Searcher, top_k


def test_top_k_is_best_first_and_breaks_ties_by_position():
    scores = np.array([1.0, 3.0, 3.0, 0.0, 2.0])
    assert top_k(scores, 3).tolist() == [1, 2, 4]
    assert top_k(scores, 10).tolist() == [1, 2, 4, 0, 3]
    assert top_k(scores, 0).tolist() == []


def test_rows_are_numbered_filing_by_filing():
    c = Corpus({"B": ["b0", "b1"], "A": ["a0"]}, {})
    assert c.docs == ["A", "B"] and len(c) == 3
    assert c.offsets == {"A": (0, 1), "B": (1, 3)}
    assert [c.key(r) for r in range(3)] == [("A", 0), ("B", 0), ("B", 1)]


def test_digest_changes_with_the_text():
    a = Corpus({"A": ["x y"]}, {}).digest()
    assert a == Corpus({"A": ["x y"]}, {}).digest()
    assert a != Corpus({"A": ["x z"]}, {}).digest()


def test_index_is_cached_and_reloaded(mini_dir):
    first = Searcher.build(mini_dir)
    assert len(list((mini_dir / "index").glob("bm25-*.npz"))) == 1
    second = Searcher.build(mini_dir)
    q = "coastal litigation settlement"
    np.testing.assert_allclose(first.bm25.scores(q.split()), second.bm25.scores(q.split()))


def test_search_finds_the_page_with_the_fact(mini_dir):
    s = Searcher.build(mini_dir)
    assert s.ranked_pages("How much did Acme pay to settle the coastal litigation?", 3)[0] == (
        "ACME_2022_10K", 1)
    hits, info = s.search("Globex green bonds", k=2)
    assert hits[0].doc == "GLOBEX_2021_10K" and hits[0].page == 1
    assert info["pages_searched"] == len(s.corpus)
