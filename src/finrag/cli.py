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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finrag")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="FinanceBench questions, metadata and filing PDFs")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--manifest", default="configs/financebench.sha256")
    p.set_defaults(func=cmd_download)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
