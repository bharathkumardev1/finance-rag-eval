import numpy as np
import pytest

from finrag.bm25 import tokenize
from finrag.chunks import ChunkedBM25, chunk, study
from finrag.data import load_questions
from finrag.search import Corpus, Searcher
from finrag.split import load_split


def test_chunks_overlap_and_cover_the_page():
    tokens = [f"t{i}" for i in range(23)]
    pieces = chunk(tokens, 10, 3)
    assert [p[0] for p in pieces] == ["t0", "t7", "t14"]
    assert pieces[-1][-1] == "t22" and all(len(p) <= 10 for p in pieces)
    assert pieces[0][-3:] == pieces[1][:3]
    assert chunk(tokens[:5], 10, 3) == [tokens[:5]] and chunk([], 10, 3) == [[]]
    with pytest.raises(ValueError):
        chunk(tokens, 10, 10)


def test_a_page_scores_as_its_best_chunk():
    page = " ".join(["filler"] * 40 + ["goodwill", "impairment"] + ["filler"] * 40)
    corpus = Corpus({"A": [page, "goodwill elsewhere"]}, {})
    index = ChunkedBM25(corpus, 16, 4)
    q = tokenize("goodwill impairment")
    best = max(index.index.scores(q)[: index.page_start[1]])
    np.testing.assert_allclose(index.scores(q)[0], best)
    assert index.scores(q, np.array([1])).shape == (1,)


def test_study_runs_pages_and_each_size(mini_dir):
    s = Searcher.build(mini_dir)
    res = study(s, load_questions(mini_dir), load_split(mini_dir / "split.json"), sizes=((16, 4),))
    configs = {r["config"] for r in res["table"]}
    assert configs == {"bm25-pages/all-pages", "bm25-pages/company_year+quarter+drop+route",
                       "bm25-chunks-16-4/all-pages", "bm25-chunks-16-4/company_year+quarter+drop+route"}
    assert res["units"]["chunks-16-4"]["documents"] > len(s.corpus)
