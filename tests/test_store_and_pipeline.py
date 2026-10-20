import json
from pathlib import Path

from finrag.bench import load_config, run_benchmark, write_markdown
from finrag.cli import main
from finrag.embedders import HashingEmbedder
from finrag.gate import check
from finrag.store import IncrementalIndex

ROOT = Path(__file__).resolve().parents[1]


def test_incremental_index_only_embeds_what_changed(tmp_path):
    calls = []
    emb = HashingEmbedder(128)
    original = emb.encode

    def counting_encode(texts, batch_size=256):
        calls.append(len(texts))
        return original(texts, batch_size)

    emb.encode = counting_encode
    idx = IncrementalIndex(emb, chunk_size=5)
    assert idx.upsert("A", ["alpha beta gamma delta epsilon zeta"]) == "added"
    assert idx.upsert("B", ["wildfire mitigation spending totaled 705 million"]) == "added"
    total = len(idx)
    calls.clear()

    assert idx.upsert("A", ["alpha beta gamma delta epsilon zeta"]) == "unchanged"
    assert calls == []  # nothing re-embedded
    assert idx.upsert("A", ["alpha beta gamma"]) == "replaced"
    assert len(idx) == total - 1  # two old chunks out, one new chunk in
    hit, _ = idx.search("wildfire mitigation", k=1)[0]
    assert hit.doc_name == "B" and hit.pages == (0,)

    idx.save(tmp_path / "idx")
    loaded = IncrementalIndex.load(tmp_path / "idx", HashingEmbedder(128))
    assert len(loaded) == len(idx)
    assert loaded.upsert("B", ["wildfire mitigation spending totaled 705 million"]) == "unchanged"
    assert loaded.remove("A") == 1 and "A" not in loaded.manifest


def test_fixture_benchmark_passes_gate_and_reports(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    cfg = load_config("configs/ci.yaml")
    report = run_benchmark(cfg, tmp_path)
    expected = len(cfg.chunkings) * len(cfg.methods) * len(cfg.filters)
    assert len(report["rows"]) == expected
    assert report["meta"]["n_questions"] == 9
    assert check(tmp_path / "benchmark.json", "configs/gate.json") == []

    write_markdown(report, None, tmp_path / "RESULTS.md")
    assert "## Retrieval grid" in (tmp_path / "RESULTS.md").read_text(encoding="utf8")


def test_gate_fails_when_a_metric_regresses(tmp_path):
    results = {"rows": [{"chunk_size": 1, "method": "bm25", "hit@10": 0.50}]}
    baseline = {"tolerance": 0.02, "rules": [
        {"match": {"chunk_size": 1, "method": "bm25"}, "min": {"hit@10": 0.60}},
        {"match": {"chunk_size": 2}, "min": {"hit@10": 0.1}},
    ]}
    (tmp_path / "r.json").write_text(json.dumps(results))
    (tmp_path / "b.json").write_text(json.dumps(baseline))
    failures = check(tmp_path / "r.json", tmp_path / "b.json")
    assert len(failures) == 2
    assert main(["gate", "--results", str(tmp_path / "r.json"),
                 "--baseline", str(tmp_path / "b.json")]) == 1
