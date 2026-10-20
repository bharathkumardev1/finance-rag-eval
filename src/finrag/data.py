"""FinanceBench download, PDF page extraction, and typed loaders.

FinanceBench (Islam et al., 2023, arXiv:2311.11944) publishes 150 open-source questions over
real SEC filings. Each question lists the evidence document and the zero-indexed page that
answers it, which is what makes page-level retrieval evaluation possible.
"""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

RAW_BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
QUESTIONS_FILE = "financebench_open_source.jsonl"
DOCS_FILE = "financebench_document_information.jsonl"


@dataclass(frozen=True)
class Evidence:
    doc_name: str
    page: int  # zero-indexed, as published by FinanceBench


@dataclass(frozen=True)
class Question:
    qid: str
    text: str
    answer: str
    company: str
    doc_name: str
    question_type: str
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class DocMeta:
    doc_name: str
    company: str
    doc_type: str
    period: int
    sector: str


def _fetch(url: str, dest: Path, timeout: int = 120) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as fh:
            while block := resp.read(1 << 16):
                fh.write(block)
        tmp.rename(dest)
        return True
    except Exception:
        tmp.unlink(missing_ok=True)
        return False


def download(data_dir: Path, workers: int = 4, all_docs: bool = True) -> dict[str, list[str]]:
    """Download the question and document files plus the filing PDFs.

    With ``all_docs`` the corpus is every FinanceBench filing (about 360), so retrieval has to
    find the right filing among many from other companies and years. Without it, only the
    filings the 150 questions reference are fetched.
    """
    data_dir = Path(data_dir)
    for name in (QUESTIONS_FILE, DOCS_FILE):
        if not _fetch(f"{RAW_BASE}/data/{name}", data_dir / name):
            raise RuntimeError(f"Could not download {name}")
    metas = load_doc_meta(data_dir)
    if all_docs:
        names = sorted(metas)
    else:
        names = sorted({q.doc_name for q in load_questions(data_dir)})

    def get(name: str) -> tuple[str, bool]:
        return name, _fetch(f"{RAW_BASE}/pdfs/{name}.pdf", data_dir / "pdfs" / f"{name}.pdf")

    ok, missing = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name, success in pool.map(get, names):
            (ok if success else missing).append(name)
    return {"downloaded": ok, "missing": missing}


def load_questions(data_dir: Path) -> list[Question]:
    rows = [json.loads(line) for line in open(Path(data_dir) / QUESTIONS_FILE, encoding="utf8")]
    out = []
    for r in rows:
        ev = tuple(
            Evidence(e.get("evidence_doc_name") or e["doc_name"], int(e["evidence_page_num"]))
            for e in r["evidence"]
        )
        out.append(
            Question(
                qid=r["financebench_id"],
                text=r["question"],
                answer=r["answer"],
                company=r["company"],
                doc_name=r["doc_name"],
                question_type=r["question_type"],
                evidence=ev,
            )
        )
    return out


def load_doc_meta(data_dir: Path) -> dict[str, DocMeta]:
    metas = {}
    for line in open(Path(data_dir) / DOCS_FILE, encoding="utf8"):
        r = json.loads(line)
        metas[r["doc_name"]] = DocMeta(
            doc_name=r["doc_name"],
            company=r["company"],
            doc_type=str(r["doc_type"]).lower(),
            period=int(r["doc_period"]),
            sector=r.get("gics_sector", ""),
        )
    return metas


def extract_pages(pdf_path: Path) -> list[str]:
    """One string per PDF page, in page order, so evidence page numbers line up.

    PyMuPDF only, and pinned in pyproject.toml: the extracted text is the corpus, so a
    different extractor or version would quietly change every number downstream.
    """
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        return [page.get_text() for page in doc]


def build_page_cache(data_dir: Path, workers: int = 2) -> dict[str, int]:
    """Extract text from every downloaded PDF into ``pages/<doc>.json`` (cached)."""
    data_dir = Path(data_dir)
    pdfs = sorted((data_dir / "pdfs").glob("*.pdf"))
    out_dir = data_dir / "pages"
    out_dir.mkdir(parents=True, exist_ok=True)

    def work(pdf: Path) -> tuple[str, int]:
        dest = out_dir / f"{pdf.stem}.json"
        if dest.exists():
            return pdf.stem, len(json.loads(dest.read_text(encoding="utf8")))
        try:
            pages = extract_pages(pdf)
        except Exception:
            return pdf.stem, -1
        dest.write_text(json.dumps(pages), encoding="utf8")
        return pdf.stem, len(pages)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(work, pdfs))


def verify_evidence_pages(data_dir: Path) -> dict[str, int]:
    """Check that each labeled evidence page matches our extracted text for that page.

    FinanceBench ships the full text of every evidence page. A slice of it should appear on
    the extracted page with the same zero-indexed number; if it shows up one page off, the
    labels and the extraction disagree about page numbering.
    """
    import re

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    pages = load_pages(data_dir)
    counts = {"match": 0, "off_by_one": 0, "not_found": 0, "no_reference": 0, "missing_doc": 0}
    for line in open(Path(data_dir) / QUESTIONS_FILE, encoding="utf8"):
        for e in json.loads(line)["evidence"]:
            doc = e.get("evidence_doc_name") or e["doc_name"]
            page = int(e["evidence_page_num"])
            ref = norm(e.get("evidence_text_full_page") or "")
            if doc not in pages:
                counts["missing_doc"] += 1
                continue
            if len(ref) < 40:
                counts["no_reference"] += 1
                continue
            probe = ref[100:180] if len(ref) > 180 else ref[:80]
            doc_pages = pages[doc]

            def on(i: int, probe: str = probe, doc_pages: list[str] = doc_pages) -> bool:
                return 0 <= i < len(doc_pages) and probe in norm(doc_pages[i])

            if on(page):
                counts["match"] += 1
            elif on(page - 1) or on(page + 1):
                counts["off_by_one"] += 1
            else:
                counts["not_found"] += 1
    return counts


def load_pages(data_dir: Path) -> dict[str, list[str]]:
    pages_dir = Path(data_dir) / "pages"
    return {
        p.stem: json.loads(p.read_text(encoding="utf8")) for p in sorted(pages_dir.glob("*.json"))
    }
