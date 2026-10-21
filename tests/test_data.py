import json

from finrag.data import DOCS_FILE, download, load_doc_meta, load_questions, sha256_file


def test_loaders_read_the_fixture(mini_dir):
    questions = load_questions(mini_dir)
    assert len(questions) == 12
    q = questions[0]
    assert q.qid == "q01" and q.doc_name == "ACME_2022_10K"
    assert q.evidence[0].page == 1

    metas = load_doc_meta(mini_dir)
    assert metas["ACME_2022Q2_10Q"].doc_type == "10q"
    assert metas["GLOBEX_2021_10K"].periods == (2021,)


def test_a_filing_listed_twice_keeps_both_years(tmp_path):
    rows = [{"doc_name": "X_2023_AR", "company": "X", "doc_type": "10k", "doc_period": p}
            for p in (2023, 2022)]
    (tmp_path / DOCS_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert load_doc_meta(tmp_path)["X_2023_AR"].periods == (2022, 2023)


def _remote(tmp_path):
    """A fake FinanceBench checkout served over file:// with the same layout as the repo."""
    remote = tmp_path / "remote"
    (remote / "data").mkdir(parents=True)
    (remote / "pdfs").mkdir()
    (remote / "data" / "questions.jsonl").write_text('{"x": 1}\n')
    (remote / "pdfs" / "A_2022_10K.pdf").write_bytes(b"%PDF-1.4 fake")
    manifest = tmp_path / "manifest.sha256"
    manifest.write_text(
        f"{sha256_file(remote / 'data' / 'questions.jsonl')}  questions.jsonl\n"
        f"{sha256_file(remote / 'pdfs' / 'A_2022_10K.pdf')}  pdfs/A_2022_10K.pdf\n"
    )
    return remote, manifest


def test_download_checks_hashes_and_skips_good_files(tmp_path):
    remote, manifest = _remote(tmp_path)
    data = tmp_path / "data"
    first = download(data, manifest, base=remote.as_uri())
    assert sorted(first["downloaded"]) == ["pdfs/A_2022_10K.pdf", "questions.jsonl"]
    second = download(data, manifest, base=remote.as_uri())
    assert sorted(second["present"]) == ["pdfs/A_2022_10K.pdf", "questions.jsonl"]


def test_download_rejects_a_file_that_changed_upstream(tmp_path):
    remote, manifest = _remote(tmp_path)
    (remote / "pdfs" / "A_2022_10K.pdf").write_bytes(b"%PDF-1.4 edited")
    summary = download(tmp_path / "data", manifest, base=remote.as_uri())
    assert summary["mismatch"] == ["pdfs/A_2022_10K.pdf"]
    assert not (tmp_path / "data" / "pdfs" / "A_2022_10K.pdf").exists()


def _pdf(path, pages):
    import pymupdf

    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    doc.save(path)


def test_pages_come_out_in_order_and_are_cached(tmp_path):
    from finrag.data import build_page_cache, load_pages

    (tmp_path / "pdfs").mkdir()
    _pdf(tmp_path / "pdfs" / "A_2022_10K.pdf", ["cover page", "balance sheet", "exhibit"])
    assert build_page_cache(tmp_path) == {"A_2022_10K": 3}
    pages = load_pages(tmp_path)["A_2022_10K"]
    assert [p.strip() for p in pages] == ["cover page", "balance sheet", "exhibit"]
    (tmp_path / "pdfs" / "A_2022_10K.pdf").unlink()
    _pdf(tmp_path / "pdfs" / "A_2022_10K.pdf", ["changed"])
    assert build_page_cache(tmp_path) == {"A_2022_10K": 3}  # cached text wins


def test_verify_finds_a_label_that_is_one_page_off(tmp_path):
    from finrag.data import QUESTIONS_FILE, verify_evidence_pages

    (tmp_path / "pages").mkdir()
    texts = ["cover " * 40, "Total current assets 1,200 and total current liabilities 900. " * 4,
             "Exhibit index " * 40]
    (tmp_path / "pages" / "A_2022_10K.json").write_text(json.dumps(texts))
    rows = [
        {"financebench_id": "a", "evidence": [
            {"doc_name": "A_2022_10K", "evidence_page_num": 1, "evidence_text_full_page": texts[1]}]},
        {"financebench_id": "b", "evidence": [
            {"doc_name": "A_2022_10K", "evidence_page_num": 2, "evidence_text_full_page": texts[1]}]},
        {"financebench_id": "c", "evidence": [
            {"doc_name": "B_2022_10K", "evidence_page_num": 0, "evidence_text_full_page": "x"}]},
    ]
    (tmp_path / QUESTIONS_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = verify_evidence_pages(tmp_path)
    assert (counts["match"], counts["off_by_one"], counts["missing_doc"]) == (1, 1, 1)
