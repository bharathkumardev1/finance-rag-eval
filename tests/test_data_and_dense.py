import json
import os
from pathlib import Path

import numpy as np
import pytest
from helpers import HashEncoder

from finrag.data import Evidence, load_questions, verify_evidence_pages
from finrag.dense import DIM, MODEL_SHA256, embed_corpus, file_sha256

MODEL_DIR = Path(os.environ.get("FINRAG_MODEL_DIR", "models/all-MiniLM-L6-v2"))
GOLDEN = Path(__file__).parent / "fixtures" / "minilm_golden.json"


def test_verify_evidence_pages_flags_misaligned_pages(tmp_path):
    page_text = "Consolidated Statement of Cash Flows " + "purchases of property plant " * 12
    (tmp_path / "pages").mkdir()
    (tmp_path / "pages" / "D.json").write_text(json.dumps(["cover", page_text, "notes"]))
    rows = [
        {"evidence": [{"doc_name": "D", "evidence_page_num": 1, "evidence_text_full_page": page_text}]},
        {"evidence": [{"doc_name": "D", "evidence_page_num": 2, "evidence_text_full_page": page_text}]},
        {"evidence": [{"doc_name": "X", "evidence_page_num": 0, "evidence_text_full_page": page_text}]},
    ]
    (tmp_path / "financebench_open_source.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    counts = verify_evidence_pages(tmp_path)
    assert counts["match"] == 1 and counts["off_by_one"] == 1 and counts["missing_doc"] == 1


def test_load_questions_accepts_both_evidence_key_names(tmp_path):
    base = {"question": "q", "answer": "x", "company": "C", "doc_name": "D", "question_type": "t"}
    rows = [{**base, "financebench_id": "a", "evidence": [{"doc_name": "D", "evidence_page_num": 3}]},
            {**base, "financebench_id": "b",
             "evidence": [{"evidence_doc_name": "E", "evidence_page_num": "4"}]}]
    (tmp_path / "financebench_open_source.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    qs = load_questions(tmp_path)
    assert qs[0].evidence == (Evidence("D", 3),) and qs[1].evidence == (Evidence("E", 4),)


def test_embed_corpus_is_incremental(mini_dir, tmp_path):
    calls = []

    class Counting(HashEncoder):
        def encode_pages(self, pages):
            calls.append(len(pages))
            return super().encode_pages(pages)

    enc = Counting()
    first = embed_corpus(enc, mini_dir, tmp_path)
    assert first == {"embedded": 7, "skipped": 0}
    assert embed_corpus(enc, mini_dir, tmp_path) == {"embedded": 0, "skipped": 7}
    (tmp_path / "ACME_2021_10K.npz").write_bytes(b"truncated")  # interrupted write
    assert embed_corpus(enc, mini_dir, tmp_path) == {"embedded": 1, "skipped": 6}
    assert len(calls) == 8


@pytest.mark.skipif(not (MODEL_DIR / "onnx" / "model.onnx").exists(),
                    reason="MiniLM not downloaded (finrag model)")
def test_minilm_matches_golden_vectors():
    """The ONNX pipeline must reproduce embeddings recorded from the official model."""
    from finrag.dense import MiniLM

    assert file_sha256(MODEL_DIR / "onnx" / "model.onnx") == MODEL_SHA256
    golden = json.loads(GOLDEN.read_text())
    got = MiniLM(MODEL_DIR).encode(golden["texts"])
    assert got.shape == (len(golden["texts"]), DIM)
    np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(got[:, :8], np.array(golden["first8"]), atol=1e-4)
