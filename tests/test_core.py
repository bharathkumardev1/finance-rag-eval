import json

import numpy as np
import pytest

from finrag.chunking import Chunk, chunk_document
from finrag.data import DocMeta, Evidence, Question, load_questions
from finrag.embedders import HashingEmbedder, LSAEmbedder, make_embedder
from finrag.filters import MetadataFilter
from finrag.metrics import aggregate, latency_summary, score_question
from finrag.retrievers import BM25Retriever, DenseRetriever, HybridRetriever, build_faiss_index


def q(text="q", evidence=(("D", 1),)):
    return Question("id", text, "", "C", evidence[0][0], "t",
                    tuple(Evidence(d, p) for d, p in evidence))


# chunking ---------------------------------------------------------------------------------

def test_chunks_track_pages():
    pages = ["a b c", "d e f", "g h"]
    chunks = chunk_document("D", pages, size=4, overlap=0)
    assert [c.text for c in chunks] == ["a b c d", "e f g h"]
    assert [c.pages for c in chunks] == [(0, 1), (1, 2)]


def test_overlap_repeats_words_and_ends_cleanly():
    chunks = chunk_document("D", ["w1 w2 w3 w4 w5 w6"], size=4, overlap=2)
    assert [c.text for c in chunks] == ["w1 w2 w3 w4", "w3 w4 w5 w6"]


def test_chunking_rejects_bad_sizes_and_handles_empty():
    with pytest.raises(ValueError):
        chunk_document("D", ["x"], size=0)
    with pytest.raises(ValueError):
        chunk_document("D", ["x"], size=4, overlap=4)
    assert chunk_document("D", ["", "  "], size=4) == []


# metrics ----------------------------------------------------------------------------------

def test_metrics_page_level():
    chunks = [Chunk("D", "x", (0,)), Chunk("E", "x", (1,)), Chunk("D", "x", (1, 2))]
    s = score_question([0, 1, 2], chunks, q(evidence=(("D", 1), ("D", 2))), ks=(1, 3))
    assert s["hit@1"] == 0.0 and s["hit@3"] == 1.0
    assert s["recall@3"] == 1.0 and s["recall@1"] == 0.0
    assert s["mrr@3"] == pytest.approx(1 / 3)
    assert s["doc_hit@1"] == 1.0  # right filing, wrong page


def test_metrics_empty_ranking_and_aggregate():
    s = score_question([], [], q(), ks=(1, 5))
    assert s["hit@5"] == 0.0 and s["recall@5"] == 0.0 and s["mrr@5"] == 0.0
    assert aggregate([{"a": 1.0}, {"a": 0.0}]) == {"a": 0.5}
    assert latency_summary([0.001, 0.003])["p50_ms"] == pytest.approx(2.0)


# filters ----------------------------------------------------------------------------------

METAS = {
    "JNJ_2021_10K": DocMeta("JNJ_2021_10K", "Johnson & Johnson", "10k", 2021, "Health"),
    "JNJ_2022_10K": DocMeta("JNJ_2022_10K", "Johnson & Johnson", "10k", 2022, "Health"),
    "AWK_2022_10K": DocMeta("AWK_2022_10K", "American Water Works", "10k", 2022, "Utilities"),
    "AXP_2022_10K": DocMeta("AXP_2022_10K", "American Express", "10k", 2022, "Financials"),
}


def test_filter_matches_names_aliases_and_years():
    f = MetadataFilter(METAS, {"Johnson & Johnson": ["JnJ"]})
    assert f.companies_in("What was JnJ's FY2022 revenue?") == {"Johnson & Johnson"}
    assert f.companies_in("American Water Works capex") == {"American Water Works"}
    assert f.years_in("Compare FY21 with fiscal 2022") == {2021, 2022}
    assert f.allowed_docs("JnJ FY2022 margin", "company_year") == {"JNJ_2022_10K"}
    assert f.allowed_docs("JnJ FY2022 margin", "company") == {"JNJ_2021_10K", "JNJ_2022_10K"}


def test_filter_falls_back_and_returns_none_without_company():
    f = MetadataFilter(METAS, {"Johnson & Johnson": ["JnJ"]})
    assert f.allowed_docs("JnJ 2019 results", "company_year") == {"JNJ_2021_10K", "JNJ_2022_10K"}
    assert f.allowed_docs("Which segment grew fastest?", "company_year") is None
    assert f.allowed_docs("JnJ anything", "none") is None


def test_filter_prefers_longest_alias():
    f = MetadataFilter(METAS)
    assert f.companies_in("American Express card spend") == {"American Express"}


# embedders and retrievers -----------------------------------------------------------------

TEXTS = [
    "capital expenditures were 120 million in fiscal 2021",
    "wildfire mitigation spending totaled 705 million",
    "subscription revenue grew 18 percent to 1040 million",
    "the company repurchased shares under its buyback program",
]


def test_embedders_return_unit_vectors():
    for emb in (HashingEmbedder(64), make_embedder("hashing-32")):
        v = emb.encode(TEXTS)
        assert v.shape == (4, emb.dim)
        assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    lsa = LSAEmbedder(dim=2)
    with pytest.raises(RuntimeError):
        lsa.encode(TEXTS)
    lsa.fit(TEXTS * 3)
    assert lsa.encode(TEXTS).shape == (4, 2)
    with pytest.raises(ValueError):
        make_embedder("nope")


def test_bm25_dense_hybrid_and_masks():
    bm25 = BM25Retriever(TEXTS)
    assert bm25.search("wildfire mitigation", 2)[0] == 1
    emb = HashingEmbedder(256)
    dense = DenseRetriever(emb, emb.encode(TEXTS))
    assert dense.search("subscription revenue", 1) == [2]
    mask = np.array([True, False, False, True])
    assert set(bm25.search("wildfire mitigation revenue", 4, mask)) <= {0, 3}
    assert set(dense.search("subscription revenue", 4, mask)) <= {0, 3}
    hybrid = HybridRetriever(bm25, dense, depth=4)
    assert hybrid.search("buyback program shares", 1) == [3]
    assert dense.search("anything", 3, np.zeros(4, dtype=bool)) == []


def test_ann_indexes_agree_with_exact_on_small_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    _, exact = build_faiss_index(x, "flat").search(x[:5], 5)
    hnsw = build_faiss_index(x, "hnsw", M=16)
    hnsw.hnsw.efSearch = 200
    _, approx = hnsw.search(x[:5], 5)
    assert (exact[:, 0] == approx[:, 0]).all()
    ivf = build_faiss_index(x, "ivf", nlist=8)
    ivf.nprobe = 8
    _, approx = ivf.search(x[:5], 5)
    assert (exact == approx).all()
    with pytest.raises(ValueError):
        build_faiss_index(x, "lsh")


# data -------------------------------------------------------------------------------------

def test_load_questions_accepts_both_evidence_key_names(tmp_path):
    rows = [
        {"financebench_id": "a", "question": "q", "answer": "x", "company": "C",
         "doc_name": "D", "question_type": "t",
         "evidence": [{"doc_name": "D", "evidence_page_num": 3}]},
        {"financebench_id": "b", "question": "q", "answer": "x", "company": "C",
         "doc_name": "D", "question_type": "t",
         "evidence": [{"evidence_doc_name": "E", "evidence_page_num": "4"}]},
    ]
    (tmp_path / "financebench_open_source.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf8")
    qs = load_questions(tmp_path)
    assert qs[0].evidence == (Evidence("D", 3),)
    assert qs[1].evidence == (Evidence("E", 4),)
