"""FinanceBench download and loaders.

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
