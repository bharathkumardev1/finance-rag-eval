"""Write results/RESULTS.md from results/eval.json, so the markdown never drifts from the data."""

from __future__ import annotations

from pathlib import Path

from .evaluate import compare
from .generate import format_p, sign_test_p
from .metrics import metrics, per_question

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


def by_type(results: dict, split: str, configs: list[str] | None = None) -> str:
    per_q = [x for x in results["questions"].values() if x["split"] == split]
    types = sorted({x["type"] for x in per_q})
    configs = configs or [r["config"] for r in results["table"] if r["split"] == split]
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


def filter_table(results: dict) -> str:
    f = results.get("filter")
    if not f:
        return ""
    body = "\n".join(
        f"| {name} | {x['n']} | {x['company_found']} | {x['filtered']} | {x['gold_filing_kept']} | "
        f"{x['mean_filings_after']:.1f} |" for name, x in f.items())
    return ("## Filter\n\n| Split | Questions | Company recognized | Filtered | Right filing kept | "
            "Filings left (mean) |\n|---|---|---|---|---|---|\n" + body + "\n")


def misses(results: dict) -> str:
    tax = (results.get("taxonomy") or {}).get("test")
    if not tax:
        return ""
    body = "\n".join(f"| {k} | {v} | {', '.join(tax['examples'][k][:3])} |"
                     for k, v in tax["counts"].items())
    return (f"## Misses of `{tax['config']}` on test ({tax['misses']} of {tax['questions']})\n\n"
            "| Category | Count | Examples |\n|---|---|---|\n" + body + "\n")


def _source(res: dict) -> str:
    m = res["meta"]
    dirty = " with uncommitted changes" if m.get("git_dirty") else ""
    return f"From `{m['command']}` at commit `{(m['git_sha'] or 'unknown')[:7]}`{dirty} on {m['date']}."


def incremental(res: dict) -> str:
    adds = ", ".join(f"{s:.2f}" for s in res["add_batch_seconds"])
    return (f"## Incremental index\n\n{_source(res)} Every tenth filing ({res['held_out_filings']} "
            f"filings, {res['held_out_pages']:,} pages) was held out, then added in batches of "
            f"{res['batch']} with a save and reload after each, leaving {res['segments']} segments. Against "
            f"a one-pass "
            f"build, over {res['queries']} queries (every question as asked and with the company's name "
            f"dropped): largest score difference {res['max_abs_score_diff']:.1e} (tolerance "
            f"{res['tolerance']:.0e}), identical top 100 for {res['same_top100']} of {res['queries']}. "
            f"Building all {res['filings']} filings took {res['full_build_seconds']:.1f} s; each added "
            f"batch took {adds} s including the reload and save.\n")


def chunk_study(res: dict) -> str:
    units = "\n".join(f"| {name} | {u['documents']:,} |" for name, u in res["units"].items())
    lines = ["## Chunk size (BM25)\n", f"{_source(res)} BM25 over chunks that never cross a page, a "
             "page scoring as its best chunk, against BM25 over whole pages.\n",
             "| Unit | Documents indexed |\n|---|---|\n" + units + "\n",
             "Test:\n", COLUMNS + "\n".join(
                 f"| `{r['config']}` | {r['hit@1']:.2f} | {r['hit@5']:.2f} | {r['hit@10']:.2f} "
                 f"({_ci(r['hit@10_ci95'])}) | {r['mrr@10']:.3f} | {r['recall@10']:.2f} | "
                 f"{r['doc_hit@10']:.2f} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} |"
                 for r in res["table"] if r["split"] == "test") + "\n",
             comparisons(res).replace("## Paired comparisons on test\n\n", "")]
    return "\n".join(lines)


def load_test(res: dict) -> str:
    rows = "\n".join(
        f"| {r['method']} | {r['concurrency']} | {r['throughput_rps']:.1f} | {r['p50_ms']:.1f} | "
        f"{r['p95_ms']:.1f} | {r['p99_ms']:.1f} | {r['errors']} | {r['server_rss_mb']:.0f} | "
        f"{r['usd_per_million']:.2f} |" for r in res["levels"])
    return (f"## Load test\n\n{_source(res)} {res['meta'].get('server', '')}. Closed-loop clients, "
            "30 s per level, cycling through the 150 questions with the full pipeline. Cost per million "
            f"queries prices one vCPU at ${res['meta']['usd_per_vcpu_hour']:.4f} an hour (half a c7i.large "
            "on demand in us-east-1).\n\n| Method | Clients | Queries/s | p50 ms | p95 ms | p99 ms | "
            "Errors | Server RSS MB | $ per million |\n|---|---|---|---|---|---|---|---|---|\n" + rows + "\n")


def rerank(res: dict, labels_status: str | None = None) -> str:
    rows = "\n".join(f"| {r['split']} | `{r['config']}` | {r['hit@1']:.2f} | {r['hit@10']:.2f} "
                     f"({_ci(r['hit@10_ci95'])}) | {r['mrr@10']:.3f} |" for r in res["table"])
    windows = res.get("windows", 1)
    cost = (f" Scoring took {1000 * res['seconds_per_pair']:.0f} ms per (question, page) pair on "
            f"{res['meta'].get('cpu', 'this machine')}, so reranking 30 pages adds about "
            f"{30 * res['seconds_per_pair']:.1f} s to a query." if res.get("seconds_per_pair") else "")
    comps = "\n".join(
        f"| {c['split']} | `{c['a']}` | `{c['b']}` | {c['metric']} | {c['diff']:+.3f} | "
        f"{_ci(c['ci95'], '+.3f')} | {c['a_better']} | {c['b_better']} | "
        f"{format_p(sign_test_p(c['a_better'], c['b_better']))} |" for c in res["comparisons"])
    return (f"## Reranker\n\n{_source(res)} {res['meta'].get('model', '')} over the top {res['top']} of "
            f"each system below, reading {windows} window{'s' if windows != 1 else ''} of 512 tokens "
            "per page, "
            f"a page scoring as its best window.{cost}\n\n"
            f"| Split | Config | Hit@1 | Hit@10 (95% CI) | MRR@10 |\n|---|---|---|---|---|\n{rows}\n\n"
            "Paired by question, 10,000 bootstrap resamples. "
            "`+rerank` is the system with its top 30 reordered.\n\n"
            "| Split | A | B | Metric | A minus B | 95% CI | A better | B better | Sign-test p |\n"
            f"|---|---|---|---|---|---|---|---|---|\n{comps}\n")


def rerank_windows(one: dict, many: dict) -> str:
    """The headline system reranked reading one window per page against several, on the same questions."""
    base, name, w = many["base"], many["base"] + "+rerank", many["windows"]
    labels = {"base": "no rerank", "one": "1 window", "many": f"{w} windows"}
    qs = {qid: {"split": v["split"], "runs": {"base": v["runs"][base], "many": v["runs"][name],
                                              "one": one["questions"][qid]["runs"][name]}}
          for qid, v in many["questions"].items() if qid in one["questions"]}
    res, parts = {"questions": qs}, ("dev", "test")
    rows = "\n".join(
        f"| {p} | {labels[k]} | "
        f"{sum(per_question([v['runs'][k] for v in qs.values() if v['split'] == p], 'hit@10')) / n:.2f} | "
        f"{sum(per_question([v['runs'][k] for v in qs.values() if v['split'] == p], 'mrr@10')) / n:.3f} |"
        for p in parts for k in labels
        if (n := sum(v["split"] == p for v in qs.values())))
    comps = "\n".join(
        f"| {p} | {labels[a]} | {labels[b]} | {m} | {c['diff']:+.3f} | {_ci(c['ci95'], '+.3f')} | "
        f"{c['a_better']} | {c['b_better']} | {format_p(sign_test_p(c['a_better'], c['b_better']))} |"
        for p in parts for a, b in (("many", "base"), ("many", "one")) for m in ("hit@10", "mrr@10")
        for c in [compare(res, a, b, p, m)])
    secs = (f" Scoring {many['pairs_scored']:,} pages took {many['scoring_seconds']:.0f} s, "
            f"{1000 * many['seconds_per_pair']:.0f} ms per page.") if many.get("seconds_per_pair") else ""
    return (f"### Reading {w} windows of each page\n\n"
            f"The same reranker on `{base}` with each page cut into up to {w} windows of 512 tokens, "
            f"a page scoring as its best window.{secs}\n\n"
            f"| Split | System | Hit@10 | MRR@10 |\n|---|---|---|---|\n{rows}\n\n"
            "Paired by question, 10,000 bootstrap resamples; the sign test is exact.\n\n"
            "| Split | A | B | Metric | A minus B | 95% CI | A better | B better | Sign-test p |\n"
            f"|---|---|---|---|---|---|---|---|---|\n{comps}\n")


NAMES = {"gold": "gold pages", "retrieved": "reported system's top 5", "naive": "BM25 over all pages, top 5",
         "closed_book": "no pages"}


def generation(res: dict, labels_status: str | None = None) -> str:
    rows = "\n".join(
        f"| {NAMES.get(name, name)} | {v['n']} | {v['accuracy']:.2f} ({_ci(v['accuracy_ci95'])}) | "
        f"{v['declined']} | {v['incorrect']} | ${v['cost_usd']:.2f} | {v['answer_p50_s']:.1f} | "
        f"{v['answer_p95_s']:.1f} |" for name, v in res["summary"]["settings"].items())
    naive = f" and `{res['naive_retrieval']}`" if res.get("naive_retrieval") else ""
    out = (f"## Generation\n\n{_source(res)} {res['generator']} answers from five pages, judged by "
           f"{res['judge']} against FinanceBench's reference answers, on the {res['split']} split"
           f"{' (first ' + str(res['limit']) + ' questions only)' if res.get('limit') else ''}. "
           f"Retrieval: `{res['retrieval']}`{naive}. Recorded spend ${res['recorded_spend_usd']:.2f} of a "
           f"${res['cap_usd']:.2f} cap.\n\n"
           "| Pages given | Questions | Correct (95% CI) | Declined | Incorrect | Cost | p50 s | p95 s |\n"
           "|---|---|---|---|---|---|---|---|\n" + rows + "\n")
    split_rows = "\n".join(
        f"| {NAMES.get(name, name)} | {v['gold_in_pages']['n']} | {_pct(v['gold_in_pages']['accuracy'])} | "
        f"{v['gold_not_in_pages']['n']} | {_pct(v['gold_not_in_pages']['accuracy'])} |"
        for name, v in res["summary"]["settings"].items() if name != "closed_book")
    out += ("\nAccuracy by whether a gold page was among the pages given:\n\n"
            "| Pages given | Questions with a gold page | Correct | Questions without | Correct |\n"
            "|---|---|---|---|---|\n" + split_rows + "\n")
    comps = res["summary"].get("comparisons") or []
    if comps:
        body = "\n".join(
            f"| {NAMES[c['a']]} | {NAMES[c['b']]} | {c['diff']:+.3f} | {_ci(c['ci95'], '+.3f')} | "
            f"{c['a_better']} | {c['b_better']} | {format_p(c['p_sign'])} |" for c in comps)
        out += ("\nPaired by question, 10,000 bootstrap resamples; the sign test is exact and "
                f"uncorrected for the {len(comps)} comparisons.\n\n"
                "| A | B | A minus B | 95% CI | A better | B better | Sign-test p |\n"
                "|---|---|---|---|---|---|---|\n" + body + "\n")
    if res.get("judge_check", {}).get("labeled"):
        j = res["judge_check"]
        ci = (f" (95% CI {_ci(j['agreement_ci95'], '.2f')}); Cohen's kappa {j['cohens_kappa']:.2f}"
              + (f" ({_ci(j['kappa_ci95'], '.2f')})" if j.get("kappa_ci95") else ""))
        out += (f"\nThe judge agreed with the {j['labeled']} labels in `{j['labels']}` on {j['agree']} "
                f"answers, {j['agreement']:.1%}{ci}, over {j.get('questions', '?')} questions. It passed "
                f"{j['judge_says_correct_you_dont']} answers the labels failed and failed "
                f"{j['judge_says_incorrect_you_dont']} answers the labels passed.\n")
        if labels_status:
            out += f"\nHow the labels were made: {labels_status}\n"
    return out


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def labels_status(results_dir: Path) -> str | None:
    """The status line of results/generation/LABELS.md, so RESULTS.md says who made the labels."""
    f = Path(results_dir) / "generation" / "LABELS.md"
    if f.exists():
        for line in f.read_text(encoding="utf8").splitlines():
            if line.lower().startswith("status:"):
                return line.split(":", 1)[1].strip()
    return None


EXTRAS = (("incremental", incremental), ("chunks", chunk_study), ("load", load_test),
          ("rerank", rerank), ("generation", generation))


def write_markdown(results: dict, path: Path, results_dir: Path | None = None) -> None:
    m = results["meta"]
    sha = (m["git_sha"] or "unknown")[:7]
    windows = f", {m['dense_windows']:,} MiniLM windows" if m.get("dense_windows") else ""
    parts = [
        "# Results\n",
        f"Produced by `{m['command']}` at commit `{sha}`"
        f"{' with uncommitted changes' if m['git_dirty'] else ''} on {m['date']}. "
        f"FinanceBench at `{m['financebench_commit'][:7]}`: {m['filings']} filings, "
        f"{m['pages']:,} pages{windows}, {m['questions']} questions ({m['dev']} dev, {m['test']} test). "
        f"{m['cpu']}, {m['cpus']} CPUs, Python {m['python']}, numpy {m['numpy']}. Latency is "
        f"in-process wall time per query, one query at a time"
        f"{' on one thread' if (m.get('threads') or {}).get('OPENBLAS_NUM_THREADS') == '1' else ''}, "
        f"including MiniLM query encoding for dense and hybrid.\n",
    ]
    if results.get("headline"):
        parts.append(f"Headline system, picked on dev: `{results['headline']}`.\n")
    parts += [
        f"## Test ({m['test']} questions)\n",
        table(results, "test"),
        multi_page(results, "test"),
        comparisons(results),
        filter_table(results),
        misses(results),
        "## Hit@10 by question type on test\n",
        by_type(results, "test", [c for c in (results.get("headline"), "bm25/all-pages") if c]
                if results.get("headline") else None),
        f"## Dev ({m['dev']} questions)\n",
        table(results, "dev"),
    ]
    if results_dir is not None:
        import json

        status = labels_status(results_dir)
        for name, section in EXTRAS:
            extra = Path(results_dir) / f"{name}.json"
            if extra.exists():
                data = json.loads(extra.read_text())
                parts.append(section(data, status) if name in ("rerank", "generation") else section(data))
                many = Path(results_dir) / "rerank-windows3.json"
                if name == "rerank" and many.exists():
                    parts.append(rerank_windows(data, json.loads(many.read_text())))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(p for p in parts if p), encoding="utf8")
