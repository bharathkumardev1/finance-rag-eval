import gzip
import json

from finrag.data import load_questions
from finrag.demo import export
from finrag.search import Searcher
from finrag.split import load_split


def test_export_has_what_the_browser_needs(mini_dir, tmp_path):
    s = Searcher.build(mini_dir)
    info = export(s, load_questions(mini_dir), load_split(mini_dir / "split.json"), tmp_path)
    assert info["filings"] == 7 and info["parity_cases"] > 0
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["n_pages"] == len(s.corpus) and meta["avg_len"] == s.bm25.avg_len
    raw = json.loads(gzip.decompress((tmp_path / "filings" / "ACME_2021_10K.json.gz").read_bytes()))
    assert len(raw["pages"]) == len(raw["lengths"]) == len(raw["kinds"]) == 8
    assert raw["kinds"][4] == "b" and raw["idf"]["acme"] == s.bm25.idf(s.bm25.vocab["acme"])
    parity = json.loads((tmp_path / "parity.json").read_text())
    assert all(len(c["top"]) == len(c["scores"]) <= 10 for c in parity)
