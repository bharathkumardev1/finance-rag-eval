import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from finrag import service  # noqa: E402
from finrag.search import Searcher  # noqa: E402


@pytest.fixture
def client(mini_dir):
    service.app.state.searcher = Searcher.build(mini_dir)
    with TestClient(service.app) as c:
        yield c
    del service.app.state.searcher


def test_health_reports_what_is_loaded(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok" and body["methods"] == ["bm25"] and body["filings"] == 7


def test_search_filters_routes_and_explains_itself(client):
    r = client.get("/search", params={"q": "What is Acme's FY2021 working capital?", "method": "bm25"})
    body = r.json()
    assert r.status_code == 200 and body["filings"] == ["ACME_2021_10K"]
    assert body["statements"] == ["balance"] and body["config"] == "bm25/company_year+quarter+drop+route"
    assert (body["hits"][0]["doc"], body["hits"][0]["page"]) == ("ACME_2021_10K", 4)
    assert body["hits"][0]["statement"] == ["balance"] and body["hits"][0]["snippet"]


def test_bad_requests_get_clear_errors(client):
    assert client.get("/search", params={"q": "capex", "method": "sparse"}).status_code == 400
    assert client.get("/search", params={"q": "capex", "method": "hybrid"}).status_code == 503
    assert client.get("/search", params={"q": "capex", "method": "bm25", "filter": "x"}).status_code == 400
    assert client.get("/search", params={"q": "x", "method": "bm25"}).status_code == 422


def test_metrics_count_requests(client):
    client.get("/search", params={"q": "Which company retired its last coal unit?", "method": "bm25"})
    text = client.get("/metrics").text
    assert 'finrag_requests_total{method="bm25",status="200"}' in text
    assert "finrag_filter_no_company_total" in text


def test_snippet_centers_on_the_query_words():
    text = "filler " * 100 + "goodwill impairment of $45 million " + "filler " * 100
    s = service.snippet(text, "goodwill impairment")
    assert "goodwill impairment" in s and len(s) <= service.SNIPPET_CHARS + 6
