"""Command line entry point: ``finrag <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_download(args: argparse.Namespace) -> int:
    from .data import download

    summary = download(Path(args.data_dir), Path(args.manifest))
    print(json.dumps({status: len(names) for status, names in summary.items()}))
    bad = summary.get("failed", []) + summary.get("mismatch", [])
    for name in bad:
        print(f"not downloaded: {name}", file=sys.stderr)
    return 1 if bad else 0


def cmd_pages(args: argparse.Namespace) -> int:
    from .data import build_page_cache

    counts = build_page_cache(Path(args.data_dir), workers=args.workers)
    failed = sorted(doc for doc, n in counts.items() if n < 0)
    print(json.dumps({"filings": len(counts), "pages": sum(n for n in counts.values() if n > 0),
                      "failed": failed}))
    return 1 if failed else 0


def cmd_verify(args: argparse.Namespace) -> int:
    from .data import verify_evidence_pages

    counts = verify_evidence_pages(Path(args.data_dir))
    print(json.dumps(counts))
    return 0 if counts["missing_doc"] == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finrag")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="FinanceBench questions, metadata and filing PDFs")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--manifest", default="configs/financebench.sha256")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("pages", help="extract page text from every PDF into data/pages")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--workers", type=int, default=2)
    p.set_defaults(func=cmd_pages)

    p = sub.add_parser("verify", help="check labeled evidence pages against the extracted text")
    p.add_argument("--data-dir", default="data")
    p.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
