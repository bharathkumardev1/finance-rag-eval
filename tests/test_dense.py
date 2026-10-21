import json
from pathlib import Path

import numpy as np
import pytest
from helpers import MODEL_DIR, HashEncoder, needs_model

from finrag.data import sha256_file
from finrag.dense import DIM, MAX_TOKENS, MODEL_SHA256, DenseIndex, download_model, embed_corpus

GOLDEN = Path(__file__).parent / "fixtures" / "minilm_golden.json"


def test_embed_corpus_is_incremental(mini_dir, tmp_path):
    calls = []

    class Counting(HashEncoder):
        def encode_pages(self, pages):
            calls.append(len(pages))
            return super().encode_pages(pages)

    enc = Counting()
    assert embed_corpus(enc, mini_dir, tmp_path) == {"embedded": 7, "skipped": 0}
    assert embed_corpus(enc, mini_dir, tmp_path) == {"embedded": 0, "skipped": 7}
    (tmp_path / "ACME_2021_10K.npz").write_bytes(b"truncated")  # an interrupted write
    assert embed_corpus(enc, mini_dir, tmp_path) == {"embedded": 1, "skipped": 6}
    assert len(calls) == 8


def test_a_page_scores_as_its_best_window():
    vecs = np.array([[1, 0], [0, 1], [0.6, 0.8], [-1, 0]], dtype=np.float32)
    owner = np.array([0, 0, 1, 2])
    index = DenseIndex(vecs, owner, n_pages=3)
    q = np.array([0, 1], dtype=np.float32)
    np.testing.assert_allclose(index.scores(q), [1.0, 0.8, 0.0])
    np.testing.assert_allclose(index.scores(q, np.array([1, 2])), [0.8, 0.0])
    assert index.scores(q, np.array([], dtype=np.int64)).shape == (0,)


def test_a_page_without_windows_is_an_error():
    with pytest.raises(ValueError, match="1 pages have no window"):
        DenseIndex(np.ones((2, 2), dtype=np.float32), np.array([0, 0]), n_pages=2)


def test_load_lines_vectors_up_with_corpus_rows(mini_dir, tmp_path):
    from finrag.search import Corpus

    embed_corpus(HashEncoder(), mini_dir, tmp_path)
    corpus = Corpus.load(mini_dir)
    index = DenseIndex.load(tmp_path, corpus.offsets)
    row = corpus.offsets["GLOBEX_2021_10K"][0] + 1
    first_window = " ".join(corpus.texts[row].split()[:30])  # HashEncoder windows are 30 words
    q = HashEncoder().encode([first_window])[0]
    scores = index.scores(q)
    assert int(np.argmax(scores)) == row and scores[row] == pytest.approx(1.0, abs=1e-3)  # stored as float16


def test_download_refuses_weights_that_are_not_the_official_ones(tmp_path):
    src = tmp_path / "mirror"
    (src / "onnx").mkdir(parents=True)
    (src / "onnx" / "model.onnx").write_bytes(b"not the model")
    (src / "tokenizer.json").write_text("{}")
    with pytest.raises(RuntimeError, match="not the official"):
        download_model(tmp_path / "dest", base=src.as_uri())


@needs_model
def test_minilm_reproduces_the_recorded_vectors():
    from finrag.dense import MiniLM

    assert sha256_file(MODEL_DIR / "onnx" / "model.onnx") == MODEL_SHA256
    golden = json.loads(GOLDEN.read_text())
    got = MiniLM(MODEL_DIR).encode(golden["texts"])
    assert got.shape == (len(golden["texts"]), DIM)
    np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(got[:, :8], np.array(golden["first8"]), atol=1e-4)


@needs_model
def test_windows_cover_the_whole_page_and_fit_the_model():
    from finrag.dense import MiniLM

    m = MiniLM(MODEL_DIR)
    text = "Net revenue rose 4% to $3.2 billion. " * 120
    ids = m._tok.encode(text, add_special_tokens=False).ids
    wins = m.windows(text)
    assert all(len(w) <= MAX_TOKENS for w in wins)
    assert [t for w in wins for t in w[1:-1]] == ids
