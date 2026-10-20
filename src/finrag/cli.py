"""Command line: ``finrag download | pages | bench | ann | report | gate``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import bench, data, gate


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="finrag", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download FinanceBench questions and filing PDFs")
    d.add_argument("--data-dir", default="data")
    d.add_argument("--referenced-only", action="store_true",
                   help="only the filings the 150 questions cite (smaller, easier corpus)")
    d.add_argument("--workers", type=int, default=4)

    pg = sub.add_parser("pages", help="extract page text from downloaded PDFs")
    pg.add_argument("--data-dir", default="data")
    pg.add_argument("--workers", type=int, default=2)

    for name, help_text in (("bench", "run the retrieval grid"),
                            ("ann", "compare exact and approximate FAISS indexes")):
        s = sub.add_parser(name, help=help_text)
        s.add_argument("--config", default="configs/default.yaml")
        s.add_argument("--out", default="results")

    r = sub.add_parser("report", help="write RESULTS.md from benchmark.json and ann.json")
    r.add_argument("--out", default="results")

    g = sub.add_parser("gate", help="fail if results drop below the baseline")
    g.add_argument("--results", default="results/benchmark.json")
    g.add_argument("--baseline", default="configs/gate.json")

    args = p.parse_args(argv)

    if args.cmd == "download":
        res = data.download(Path(args.data_dir), args.workers, all_docs=not args.referenced_only)
        print(f"downloaded {len(res['downloaded'])} filings, missing {len(res['missing'])}")
        for name in res["missing"]:
            print(f"  missing: {name}")
    elif args.cmd == "pages":
        counts = data.build_page_cache(Path(args.data_dir), args.workers)
        failed = [k for k, v in counts.items() if v < 0]
        print(f"parsed {len(counts) - len(failed)} filings, "
              f"{sum(v for v in counts.values() if v > 0):,} pages, {len(failed)} failed")
    elif args.cmd == "bench":
        bench.run_benchmark(bench.load_config(args.config), args.out)
    elif args.cmd == "ann":
        bench.run_ann(bench.load_config(args.config), args.out)
    elif args.cmd == "report":
        out = Path(args.out)
        b = json.loads((out / "benchmark.json").read_text(encoding="utf8"))
        a_path = out / "ann.json"
        a = json.loads(a_path.read_text(encoding="utf8")) if a_path.exists() else None
        bench.write_markdown(b, a, out / "RESULTS.md")
        print(f"wrote {out / 'RESULTS.md'}")
    elif args.cmd == "gate":
        failures = gate.check(args.results, args.baseline)
        if failures:
            print("quality gate FAILED:")
            for f in failures:
                print(f"  {f}")
            return 1
        print("quality gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
