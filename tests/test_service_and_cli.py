import json

import pytest

from finrag.cli import main
from finrag.gate import check
from finrag.service import snippet

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(searcher):
    from fastapi.testclient import TestClient

    from finrag.service import app

    app.state.searcher = searcher
    with TestClient(app) as c:
        yield c


def test_health_and_search(client):
    h = client.get("/healthz").json()
    assert h["status"] == "ok" and h["methods"] == ["bm25"] and h["filings"] == 7
    r = client.get("/search", params={"q": "Globex wildfire mitigation spending FY2022",
                                      "method": "bm25", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["filings"] == ["GLOBEX_2022_10K"] and len(body["hits"]) == 2
    assert body["hits"][0]["page"] == 1 and "705" in body["hits"][0]["snippet"]


def test_bad_requests_get_clear_errors(client):
    assert client.get("/search", params={"q": "x"}).status_code == 422  # too short
    assert client.get("/search", params={"q": "revenue", "method": "magic"}).status_code == 400
    assert client.get("/search", params={"q": "revenue", "method": "dense"}).status_code == 503
    assert client.get("/search", params={"q": "revenue", "method": "bm25",
                                         "filter": "zip"}).status_code == 400


def test_metrics_endpoint_counts_requests(client):
    client.get("/search", params={"q": "Which company retired its last coal unit?",
                                  "method": "bm25"})
    text = client.get("/metrics").text
    assert 'finrag_requests_total{method="bm25",status="200"}' in text
    assert "finrag_filter_no_company_total" in text and "finrag_search_seconds_bucket" in text


def test_snippet_centers_on_query_words():
    text = "filler " * 200 + "the coastal litigation settled for $80 million " + "filler " * 200
    s = snippet(text, "coastal litigation settlement")
    assert "coastal litigation" in s and len(s) <= 282
    assert snippet("short page", "anything") == "short page"


def test_fixture_eval_passes_the_gate(mini_dir, tmp_path):
    out = tmp_path / "res"
    assert main(["eval", "--data-dir", str(mini_dir), "--split", str(mini_dir / "split.json"),
                 "--out", str(out), "--no-dense", "--aliases", "missing.yaml"]) == 0
    assert check(out / "eval.json", ROOT / "configs" / "gate.json") == []
    report = (out / "RESULTS.md").read_text()
    assert "## Test split" in report and "Paired comparisons" in report


def test_gate_fails_on_regression(tmp_path):
    results = {"table": [{"config": "bm25/none", "split": "all", "hit@5": 0.5}]}
    baseline = {"tolerance": 0.0, "rules": [
        {"match": {"config": "bm25/none", "split": "all"}, "min": {"hit@5": 0.9}},
        {"match": {"config": "missing", "split": "all"}, "min": {"hit@5": 0.1}}]}
    (tmp_path / "r.json").write_text(json.dumps(results))
    (tmp_path / "b.json").write_text(json.dumps(baseline))
    assert len(check(tmp_path / "r.json", tmp_path / "b.json")) == 2
    assert main(["gate", "--results", str(tmp_path / "r.json"),
                 "--baseline", str(tmp_path / "b.json")]) == 1
