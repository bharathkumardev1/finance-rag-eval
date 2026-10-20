from pathlib import Path

import pytest
from helpers import HashEncoder

ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "tests" / "fixtures" / "mini"


@pytest.fixture(scope="session")
def mini_dir(tmp_path_factory):
    """A copy of the fixture corpus, so index caches don't land in the repo."""
    import shutil

    dest = tmp_path_factory.mktemp("mini")
    shutil.copytree(MINI, dest, dirs_exist_ok=True)
    return dest


@pytest.fixture(scope="session")
def searcher(mini_dir):
    from finrag.search import Searcher

    return Searcher.build(mini_dir, {})


@pytest.fixture(scope="session")
def dense_searcher(mini_dir, tmp_path_factory):
    from finrag.dense import DenseIndex, embed_corpus
    from finrag.search import Searcher

    emb = tmp_path_factory.mktemp("emb")
    enc = HashEncoder()
    embed_corpus(enc, mini_dir, emb)
    base = Searcher.build(mini_dir, {})
    dense = DenseIndex.load(emb, base.corpus.offsets)
    return Searcher(base.corpus, base.bm25, base.filter, dense, enc)
