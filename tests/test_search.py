import numpy as np
import pytest

from finrag.search import RRF_DEPTH, Corpus, Searcher, rrf, top_k


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


def test_rrf_rewards_pages_both_lists_rank_high():
    rows, scores = rrf([np.array([5, 7, 9]), np.array([7, 5, 8])], k=60)
    assert rows.tolist() == [5, 7, 8, 9]  # 5 and 7 tie on 1/61 + 1/62; ties go to the lower row
    assert scores[0] == scores[1] == 1 / 61 + 1 / 62 and scores[2] == scores[3] == 1 / 63


def hash_searcher(mini_dir, tmp_path):
    from helpers import HashEncoder

    from finrag.dense import DenseIndex, embed_corpus

    embed_corpus(HashEncoder(), mini_dir, tmp_path)
    s = Searcher.build(mini_dir)
    return Searcher(s.corpus, s.bm25, DenseIndex.load(tmp_path, s.corpus.offsets), HashEncoder())


def test_hybrid_fuses_the_bm25_and_dense_rankings(mini_dir, tmp_path):
    s = hash_searcher(mini_dir, tmp_path)
    assert s.methods == ("bm25", "dense", "hybrid")
    q = "Globex green bonds due 2031"
    bm25 = s.rank(q, RRF_DEPTH, "bm25")[0].tolist()
    dense = s.rank(q, RRF_DEPTH, "dense")[0].tolist()
    expected, _ = rrf([np.array(bm25), np.array(dense)])
    assert s.rank(q, 10, "hybrid")[0].tolist() == expected[:10].tolist()
    assert s.ranked_pages(q, 1, "hybrid") == [("GLOBEX_2021_10K", 1)]


def test_dense_needs_an_index():
    s = Searcher(Corpus({"A": ["x"]}, {}), None)
    assert s.methods == ("bm25",)
    with pytest.raises(RuntimeError, match="no dense index"):
        s.rank("x", 1, "dense")
    with pytest.raises(ValueError, match="unknown method"):
        s.rank("x", 1, "sparse")
