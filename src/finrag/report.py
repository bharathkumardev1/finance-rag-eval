"""Write results/RESULTS.md from results/eval.json, so the markdown never drifts from the data."""

from __future__ import annotations

from pathlib import Path

from .metrics import metrics

COLUMNS = ("| Config | Hit@1 | Hit@5 | Hit@10 (95% CI) | MRR@10 | Recall@10 | Right filing@10 "
           "| p50 ms | p95 ms |\n|---|---|---|---|---|---|---|---|---|\n")


def _ci(ci: list[float], fmt: str = ".2f") -> str:
    return f"{ci[0]:{fmt}} to {ci[1]:{fmt}}"


def table(results: dict, split: str) -> str:
    lines = [
        f"| `{r['config']}` | {r['hit@1']:.2f} | {r['hit@5']:.2f} | {r['hit@10']:.2f} "
        f"({_ci(r['hit@10_ci95'])}) | {r['mrr@10']:.3f} | {r['recall@10']:.2f} | "
        f"{r['doc_hit@10']:.2f} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} |"
        for r in results["table"] if r["split"] == split
    ]
    return COLUMNS + "\n".join(lines) + "\n"


def multi_page(results: dict, split: str) -> str:
    rows = [r for r in results["table"] if r["split"] == split and r.get("n_multi_page")]
    if not rows:
        return ""
    n = rows[0]["n_multi_page"]
    body = "\n".join(f"| `{r['config']}` | {r['recall@10_multi_page']:.2f} |" for r in rows)
    return (f"{n} of the {rows[0]['n']} questions have more than one gold page. Recall@10 on "
            f"those alone:\n\n| Config | Recall@10, multi-page |\n|---|---|\n{body}\n")


def by_type(results: dict, split: str) -> str:
    per_q = [x for x in results["questions"].values() if x["split"] == split]
    types = sorted({x["type"] for x in per_q})
    configs = [r["config"] for r in results["table"] if r["split"] == split]
    head = "| Config | " + " | ".join(f"{t} Hit@10" for t in types) + " |\n"
    head += "|---|" + "---|" * len(types) + "\n"
    lines = []
    for c in configs:
        cells = []
        for t in types:
            recs = [x["runs"][c] for x in per_q if x["type"] == t]
            cells.append(f"{metrics(recs)['hit@10']:.2f} (n={len(recs)})")
        lines.append(f"| `{c}` | " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"


def comparisons(results: dict) -> str:
    rows = results.get("comparisons") or []
    if not rows:
        return ""
    body = "\n".join(
        f"| `{c['a']}` | `{c['b']}` | {c['metric']} | {c['diff']:+.3f} | {_ci(c['ci95'], '+.3f')} | "
        f"{c['a_better']} | {c['b_better']} |" for c in rows)
    return ("## Paired comparisons on test\n\nSame questions, 10,000 bootstrap resamples of the "
            "per-question difference.\n\n| A | B | Metric | A minus B | 95% CI | A better | "
            "B better |\n|---|---|---|---|---|---|---|\n" + body + "\n")


def write_markdown(results: dict, path: Path) -> None:
    m = results["meta"]
    sha = (m["git_sha"] or "unknown")[:7]
    parts = [
        "# Results\n",
        f"Produced by `{m['command']}` at commit `{sha}`"
        f"{' with uncommitted changes' if m['git_dirty'] else ''} on {m['date']}. "
        f"FinanceBench at `{m['financebench_commit'][:7]}`: {m['filings']} filings, "
        f"{m['pages']:,} pages, {m['questions']} questions ({m['dev']} dev, {m['test']} test). "
        f"{m['cpu']}, {m['cpus']} CPUs, Python {m['python']}, numpy {m['numpy']}. Latency is "
        f"in-process wall time per query, one query at a time.\n",
        f"## Test ({m['test']} questions)\n",
        table(results, "test"),
        multi_page(results, "test"),
        "By question type:\n",
        by_type(results, "test"),
        comparisons(results),
        f"## Dev ({m['dev']} questions)\n",
        table(results, "dev"),
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(p for p in parts if p), encoding="utf8")
