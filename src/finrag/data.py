"""FinanceBench download, page text extraction and loaders.

FinanceBench (Islam et al., 2023, arXiv:2311.11944) publishes 150 questions over real SEC
filings. Each question names the filing and the zero-indexed page that answers it, which is
what makes page-level retrieval evaluation possible.

Every file comes from one commit of the FinanceBench repo and is checked against
configs/financebench.sha256, so an upstream change fails the download instead of quietly
moving every number downstream.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

FINANCEBENCH_COMMIT = "cc39aeb4afdf33909ee1412188bf89035950c2eb"
RAW_BASE = f"https://raw.githubusercontent.com/patronus-ai/financebench/{FINANCEBENCH_COMMIT}"
QUESTIONS_FILE = "financebench_open_source.jsonl"
DOCS_FILE = "financebench_document_information.jsonl"


@dataclass(frozen=True)
class Evidence:
    doc_name: str
    page: int  # zero-indexed, as published


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
    periods: tuple[int, ...]  # fiscal years the metadata lists for this filing
    sector: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path: Path) -> dict[str, str]:
    """``sha256sum`` format: one ``<hex>  <relative path>`` line per file."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    return out


def _fetch(url: str, dest: Path, expected: str, timeout: int = 180) -> str:
    """Download ``url`` to ``dest`` unless a file with the expected hash is already there."""
    if dest.exists() and sha256_file(dest) == expected:
        return "present"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as fh:
            while block := resp.read(1 << 16):
                fh.write(block)
    except OSError:
        tmp.unlink(missing_ok=True)
        return "failed"
    if sha256_file(tmp) != expected:
        tmp.unlink()
        return "mismatch"
    tmp.replace(dest)
    return "downloaded"


def download(data_dir: Path, manifest: Path, base: str = RAW_BASE, workers: int = 6) -> dict:
    """Fetch every file the manifest lists: the two question/metadata files and 360 PDFs."""
    data_dir = Path(data_dir)
    files = read_manifest(manifest)

    def get(item: tuple[str, str]) -> tuple[str, str]:
        name, digest = item
        remote = name if name.startswith("pdfs/") else f"data/{name}"
        return name, _fetch(f"{base}/{remote}", data_dir / name, digest)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = dict(pool.map(get, sorted(files.items())))
    summary: dict[str, list[str]] = {}
    for name, status in results.items():
        summary.setdefault(status, []).append(name)
    return summary


def load_questions(data_dir: Path) -> list[Question]:
    out = []
    for line in open(Path(data_dir) / QUESTIONS_FILE, encoding="utf8"):
        r = json.loads(line)
        evidence = tuple(
            Evidence(e.get("evidence_doc_name") or e["doc_name"], int(e["evidence_page_num"]))
            for e in r["evidence"]
        )
        out.append(Question(
            qid=r["financebench_id"],
            text=r["question"],
            answer=r["answer"],
            company=r["company"],
            doc_name=r["doc_name"],
            question_type=r["question_type"],
            evidence=evidence,
        ))
    return out


def load_doc_meta(data_dir: Path) -> dict[str, DocMeta]:
    """Filing metadata keyed by name.

    One filing, FOOTLOCKER_2023_annualreport, is listed twice, once as 2022 and once as 2023.
    No question cites it, and its fiscal year ends in January 2023, so it keeps both years.
    """
    rows: dict[str, list[dict]] = {}
    for line in open(Path(data_dir) / DOCS_FILE, encoding="utf8"):
        r = json.loads(line)
        rows.setdefault(r["doc_name"], []).append(r)
    metas = {}
    for name, rs in rows.items():
        metas[name] = DocMeta(
            doc_name=name,
            company=rs[0]["company"],
            doc_type=str(rs[0]["doc_type"]).lower(),
            periods=tuple(sorted({int(r["doc_period"]) for r in rs})),
            sector=rs[0].get("gics_sector", ""),
        )
    return metas


def extract_pages(pdf_path: Path) -> list[str]:
    """One string per PDF page, in page order, so FinanceBench's page numbers line up."""
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        return [page.get_text() for page in doc]


def build_page_cache(data_dir: Path, workers: int = 2) -> dict[str, int]:
    """Extract every downloaded PDF into ``pages/<filing>.json``; existing files are kept."""
    data_dir = Path(data_dir)
    out_dir = data_dir / "pages"
    out_dir.mkdir(parents=True, exist_ok=True)

    def work(pdf: Path) -> tuple[str, int]:
        dest = out_dir / f"{pdf.stem}.json"
        if dest.exists():
            return pdf.stem, len(json.loads(dest.read_text(encoding="utf8")))
        try:
            pages = extract_pages(pdf)
        except Exception:  # a PDF PyMuPDF can't open is reported, not fatal
            return pdf.stem, -1
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_text(json.dumps(pages), encoding="utf8")
        tmp.replace(dest)
        return pdf.stem, len(pages)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(work, sorted((data_dir / "pdfs").glob("*.pdf"))))


def load_pages(data_dir: Path) -> dict[str, list[str]]:
    pages_dir = Path(data_dir) / "pages"
    return {p.stem: json.loads(p.read_text(encoding="utf8")) for p in sorted(pages_dir.glob("*.json"))}


def verify_evidence_pages(data_dir: Path) -> dict[str, int]:
    """Check each labeled evidence page against the text we extracted for that page number.

    FinanceBench ships the full text of every evidence page. An 80-character slice of it
    (letters and digits only) should appear on our page with the same number. If it shows up
    one page away instead, the labels and the extraction disagree about numbering.
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
            text = [norm(t) for t in pages[doc][max(page - 1, 0):page + 2]]
            here = page - max(page - 1, 0)
            if here < len(text) and probe in text[here]:
                counts["match"] += 1
            elif any(probe in t for i, t in enumerate(text) if i != here):
                counts["off_by_one"] += 1
            else:
                counts["not_found"] += 1
    return counts
