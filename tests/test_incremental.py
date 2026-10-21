import json
import shutil

import numpy as np

from finrag.bm25 import tokenize
from finrag.data import load_questions
from finrag.filters import MetadataFilter
from finrag.incremental import TOLERANCE, check
from finrag.search import Corpus, load_or_build_bm25


def test_a_new_filing_is_added_as_a_segment_without_a_rebuild(mini_dir, tmp_path):
    data = tmp_path / "data"
    shutil.copytree(mini_dir, data)
    held = data / "pages" / "INITECH_2022_10K.json"
    pages = json.loads(held.read_text())
    held.unlink()
    first = load_or_build_bm25(Corpus.load(data), data / "index")
    assert len(first.segments) == 1 and "INITECH_2022_10K" not in first.filings()
    segment_file = data / "index" / "bm25" / "segment-0000.npz"
    before = segment_file.stat().st_mtime_ns

    held.write_text(json.dumps(pages))  # the new filing arrives
    corpus = Corpus.load(data)
    grown = load_or_build_bm25(corpus, data / "index")
    assert len(grown.segments) == 2 and segment_file.stat().st_mtime_ns == before
    fresh = load_or_build_bm25(corpus, tmp_path / "fresh-index")
    q = tokenize("Initech restructuring charges headcount 2022")
    np.testing.assert_array_equal(grown.scores(q), fresh.scores(q))


def test_a_changed_filing_forces_a_rebuild(mini_dir, tmp_path):
    data = tmp_path / "data"
    shutil.copytree(mini_dir, data)
    load_or_build_bm25(Corpus.load(data), data / "index")
    path = data / "pages" / "ACME_2021_10K.json"
    pages = json.loads(path.read_text())
    pages[1] = "Acme re-extracted this page."
    path.write_text(json.dumps(pages))
    index = load_or_build_bm25(Corpus.load(data), data / "index")
    assert len(index.segments) == 1 and index.scores(["re"]).max() > 0


def test_check_passes_on_the_fixture(mini_dir):
    corpus = Corpus.load(mini_dir)
    result = check(corpus, load_questions(mini_dir), MetadataFilter(corpus.metas), holdout_every=3,
                   batch=1)
    assert result["passed"] and result["segments"] == 1 + result["held_out_filings"] == 4
    assert result["max_abs_score_diff"] <= TOLERANCE and result["same_top100"] == result["queries"]
