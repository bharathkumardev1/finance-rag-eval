"""Answer generation on the test split, scored apart from retrieval.

A generator model answers each test question in up to four settings:

* ``retrieved``: the top 5 pages of the reported retrieval system, read from results/eval.json,
  so this step never reruns retrieval or needs the MiniLM embeddings.
* ``gold``: the question's gold pages, so the gap between ``gold`` and ``retrieved`` is what
  retrieval costs and what's left in ``gold`` is the generator's own error.
* ``naive``: the top 5 pages of BM25 over every page, with no filter or routing, which is
  what the reported system has to beat end to end.
* ``closed_book``: no pages at all. FinanceBench is public, so this is how much the model
  already knows, and it bounds how much of the other settings' accuracy the pages earn.

A stronger judge model compares each answer with FinanceBench's reference answer and says
correct or incorrect. An answer of exactly ``NOT IN PAGES`` is scored as declined without
asking the judge. ``finrag label-judge`` samples judged answers for a person to label, and
``finrag judge-agreement`` measures how often the judge agrees with that person.

Every API call goes through a recording keyed by a hash of the request, with the model the
API reports, the tokens and the cost. A rerun replays the recording, and ``--replay`` makes
a run that needs no API key. Spending is checked before each call against a worst-case
estimate, and the run stops before it can pass the cap. The cap covers everything in the
recording, not just one run. A reply cut off at its token allowance stops the run too: it's
recorded as incomplete so its cost counts, and never replayed, so a rerun asks again.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data import Question
from .metrics import bootstrap_ci, gold_pages, paired_diff

# USD per million tokens, from developers.openai.com/api/docs/pricing on 2026-10-21.
PRICES = {
    "gpt-6-luna": {"input": 0.10, "cached": 0.01, "output": 0.50},
    "gpt-6.1-sol": {"input": 2.00, "cached": 0.10, "output": 10.00},
}
GENERATOR = "gpt-6-luna"
JUDGE = "gpt-6.1-sol"
CAP_USD = 25.0
TOP_PAGES = 5
SETTINGS = ("gold", "retrieved", "naive", "closed_book")
COMPARISONS = (("gold", "retrieved"), ("retrieved", "naive"), ("retrieved", "closed_book"),
               ("gold", "closed_book"))
DECLINE = "NOT IN PAGES"
# Reasoning tokens count against these, and a cut-off reply stops the run. Raising them after
# a run would change every request's hash and void the recording, so they start generous;
# only tokens actually produced are billed.
MAX_ANSWER_TOKENS = 8000
MAX_JUDGE_TOKENS = 2000
CHARS_PER_TOKEN = 3  # worst case for the pre-call estimate; English filings run about 4

ANSWER_INSTRUCTIONS = (
    "You answer questions about public company filings using only the filing pages provided. "
    "Give the figure or fact asked for, with units and the fiscal period, then one sentence "
    "saying which page it comes from and how you computed it if you did. If the pages don't "
    f"contain what you need, reply with exactly {DECLINE} and nothing else."
)
CLOSED_BOOK_INSTRUCTIONS = (
    "You answer questions about public company filings from memory; no documents are provided. "
    "Give the figure or fact asked for, with units and the fiscal period, then one sentence "
    f"on how you know it or computed it. If you don't know it, reply with exactly {DECLINE} "
    "and nothing else."
)
JUDGE_INSTRUCTIONS = (
    "You grade an answer to a question about a company filing against a reference answer "
    "written by a financial analyst. The answer is correct if it gives the same figure or "
    "conclusion as the reference: numbers may differ by rounding or by up to 1%, and units "
    "or wording may differ. It is incorrect if the figure or conclusion differs, if it hedges "
    "between several values, or if it answers a different question. Extra explanation is fine."
)
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["correct", "incorrect"]},
                   "reason": {"type": "string"}},
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}


@dataclass
class Reply:
    text: str
    model: str            # the model the API says answered, which can be more specific
    input_tokens: int
    cached_tokens: int
    output_tokens: int    # includes reasoning tokens, which are billed as output
    reasoning_tokens: int
    seconds: float
    status: str


class BudgetExceeded(RuntimeError):
    pass


class MissingRecording(RuntimeError):
    pass


class IncompleteReply(RuntimeError):
    """The reply hit its max_output_tokens; it's billed and recorded as incomplete."""


def price(model: str) -> dict[str, float]:
    for name in sorted(PRICES, key=len, reverse=True):  # a dated snapshot bills as its model
        if model == name or model.startswith(name + "-"):
            return PRICES[name]
    raise KeyError(f"no price for {model!r}; add it to PRICES before spending on it")


def cost(reply: Reply) -> float:
    p = price(reply.model)
    fresh = reply.input_tokens - reply.cached_tokens
    return (fresh * p["input"] + reply.cached_tokens * p["cached"] + reply.output_tokens * p["output"]) / 1e6


class Budget:
    def __init__(self, cap_usd: float):
        self.cap = cap_usd
        self.spent = 0.0

    def worst_case(self, request: dict) -> float:
        p = price(request["model"])
        chars = len(request["instructions"]) + len(request["input"])
        return (chars / CHARS_PER_TOKEN * p["input"] + request["max_output_tokens"] * p["output"]) / 1e6

    def check(self, request: dict) -> None:
        if self.spent + self.worst_case(request) > self.cap:
            raise BudgetExceeded(f"spent ${self.spent:.4f}; the next call could pass the ${self.cap:.2f} cap")


def openai_client() -> Callable[[dict], Reply]:
    from openai import OpenAI

    client = OpenAI(max_retries=3, timeout=120)

    def call(request: dict) -> Reply:
        t0 = time.perf_counter()
        r = client.responses.create(**request, store=False)
        u = r.usage
        return Reply(text=r.output_text, model=r.model, input_tokens=u.input_tokens,
                     cached_tokens=(u.input_tokens_details.cached_tokens or 0),
                     output_tokens=u.output_tokens,
                     reasoning_tokens=(u.output_tokens_details.reasoning_tokens or 0),
                     seconds=round(time.perf_counter() - t0, 3), status=r.status)

    return call


class Recorder:
    """Calls by a hash of the request, appended to a JSONL file and replayed from it."""

    def __init__(self, path: Path, client: Callable[[dict], Reply] | None, budget: Budget):
        self.path = Path(path)
        self.client = client
        self.budget = budget
        self.calls: dict[str, dict] = {}
        self.incomplete: list[dict] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf8").splitlines():
                rec = json.loads(line)
                if rec.get("incomplete"):
                    self.incomplete.append(rec)
                else:
                    self.calls[rec["key"]] = rec
        self.replayed = self.made = 0
        self.budget.spent = sum(rec.get("cost_usd", 0.0) for rec in [*self.calls.values(), *self.incomplete])

    @staticmethod
    def key(request: dict) -> str:
        return hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf8")).hexdigest()

    def call(self, request: dict, tag: dict) -> tuple[Reply, bool]:
        """The reply, and whether it came from the recording."""
        k = self.key(request)
        if k in self.calls:
            self.replayed += 1
            return Reply(**self.calls[k]["reply"]), True
        if self.client is None:
            raise MissingRecording(f"no recorded reply for {tag}; run without --replay to make the call")
        self.budget.check(request)
        reply = self.client(request)
        self.budget.spent += cost(reply)
        rec = {"key": k, "tag": tag, "reply": asdict(reply), "cost_usd": cost(reply)}
        if reply.status != "completed":
            rec["incomplete"] = True
            self.incomplete.append(rec)
            self._append(rec)
            raise IncompleteReply(f"{tag}: response {reply.status} after {reply.output_tokens} output tokens "
                                  f"({reply.reasoning_tokens} reasoning); rerun to ask again")
        self.calls[k] = rec
        self._append(rec)
        self.made += 1
        return reply, False

    def _append(self, rec: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf8") as fh:
            fh.write(json.dumps(rec) + "\n")


def answer_request(q: Question, pages: list[tuple[str, int, str]], model: str) -> dict:
    if not pages:  # closed book
        return {"model": model, "instructions": CLOSED_BOOK_INSTRUCTIONS, "input": f"Question: {q.text}",
                "max_output_tokens": MAX_ANSWER_TOKENS, "reasoning": {"effort": "medium"}}
    blocks = [f"[{doc}, page {page + 1}]\n{text.strip()}" for doc, page, text in pages]
    return {"model": model, "instructions": ANSWER_INSTRUCTIONS,
            "input": "Filing pages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {q.text}",
            "max_output_tokens": MAX_ANSWER_TOKENS, "reasoning": {"effort": "medium"}}


def judge_request(q: Question, answer: str, model: str) -> dict:
    return {"model": model, "instructions": JUDGE_INSTRUCTIONS,
            "input": f"Question: {q.text}\n\nReference answer: {q.answer}\n\nAnswer to grade: {answer}",
            "max_output_tokens": MAX_JUDGE_TOKENS, "reasoning": {"effort": "low"},
            "text": {"format": {"type": "json_schema", "name": "verdict", "schema": VERDICT_SCHEMA,
                                "strict": True}}}


def page_sets(q: Question, retrieved: list[list], naive: list[list] | None = None
              ) -> dict[str, list[tuple[str, int]]]:
    sets = {"retrieved": [(d, p) for d, p in retrieved[:TOP_PAGES]],
            "gold": sorted(gold_pages(q))[:TOP_PAGES], "closed_book": []}
    if naive is not None:
        sets["naive"] = [(d, p) for d, p in naive[:TOP_PAGES]]
    return sets


def run(questions: list[Question], retrieved: dict[str, list[list]], page_text: Callable[[str, int], str],
        recorder: Recorder, generator: str = GENERATOR, judge: str = JUDGE,
        naive: dict[str, list[list]] | None = None,
        settings: tuple[str, ...] = ("retrieved", "gold")) -> dict:
    """Answer and judge every question in each setting; stops cleanly at the budget cap or a
    cut-off reply, keeping the rows it finished."""
    rows = []
    stopped = None
    try:
        for q in questions:
            sets = page_sets(q, retrieved[q.qid], None if naive is None else naive[q.qid])
            gold = {tuple(k) for k in gold_pages(q)}
            for setting in settings:
                keys = sets[setting]
                pages = [(d, p, page_text(d, p)) for d, p in keys]
                tag = {"qid": q.qid, "setting": setting, "step": "answer"}
                ans, _ = recorder.call(answer_request(q, pages, generator), tag)
                row = {"qid": q.qid, "type": q.question_type, "setting": setting,
                       "pages": [list(k) for k in keys], "gold_in_pages": bool(gold & set(keys)),
                       "answer": ans.text, "answer_model": ans.model,
                       "answer_status": ans.status, "answer_seconds": ans.seconds,
                       "cost_usd": cost(ans)}
                if ans.text.strip() == DECLINE:
                    row.update(verdict="declined", reason="")
                else:
                    tag = {**tag, "step": "judge"}
                    jr, _ = recorder.call(judge_request(q, ans.text, judge), tag)
                    v = json.loads(jr.text)
                    row.update(verdict=v["verdict"], reason=v["reason"], judge_model=jr.model)
                    row["cost_usd"] += cost(jr)
                rows.append(row)
    except (BudgetExceeded, IncompleteReply) as e:
        stopped = str(e)
    return {"generator": generator, "judge": judge, "top_pages": TOP_PAGES, "settings": list(settings),
            "rows": rows,
            "summary": summarize(rows), "stopped": stopped,
            "calls": {"made": recorder.made, "replayed": recorder.replayed},
            "recorded_spend_usd": round(recorder.budget.spent, 6), "cap_usd": recorder.budget.cap}


def format_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def sign_test_p(a_better: int, b_better: int) -> float:
    """Exact two-sided sign test on the questions where two settings differ."""
    n = a_better + b_better
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(a_better, b_better) + 1))
    return min(1.0, 2 * tail / 2**n)


def _accuracy(rs: list[dict]) -> dict:
    return {"n": len(rs), "accuracy": float(np.mean([r["verdict"] == "correct" for r in rs])) if rs else None}


def summarize(rows: list[dict]) -> dict:
    out: dict = {"settings": {}, "comparisons": []}
    for setting in SETTINGS:
        rs = [r for r in rows if r["setting"] == setting]
        if not rs:
            continue
        correct = np.array([r["verdict"] == "correct" for r in rs], dtype=float)
        secs = np.array([r["answer_seconds"] for r in rs])
        out["settings"][setting] = {
            "n": len(rs), "accuracy": float(correct.mean()), "accuracy_ci95": list(bootstrap_ci(correct)),
            "declined": sum(r["verdict"] == "declined" for r in rs),
            "incorrect": sum(r["verdict"] == "incorrect" for r in rs),
            "cost_usd": round(float(sum(r["cost_usd"] for r in rs)), 10),
            "answer_p50_s": float(np.percentile(secs, 50)), "answer_p95_s": float(np.percentile(secs, 95)),
            "by_type": {t: float(np.mean([r["verdict"] == "correct" for r in rs if r["type"] == t]))
                        for t in sorted({r["type"] for r in rs})},
            "gold_in_pages": _accuracy([r for r in rs if r.get("gold_in_pages")]),
            "gold_not_in_pages": _accuracy([r for r in rs if not r.get("gold_in_pages")]),
        }
    by_q: dict[str, dict[str, float]] = {}
    for r in rows:
        by_q.setdefault(r["qid"], {})[r["setting"]] = float(r["verdict"] == "correct")
    for a, b in COMPARISONS:
        paired = [(v[a], v[b]) for v in by_q.values() if a in v and b in v]
        if paired:
            va, vb = np.array(paired).T
            d = paired_diff(va, vb)
            p = sign_test_p(d["a_better"], d["b_better"])
            out["comparisons"].append({"a": a, "b": b, **d, "p_sign": p})
    first = next((c for c in out["comparisons"] if (c["a"], c["b"]) == ("gold", "retrieved")), None)
    if first:
        out["gold_minus_retrieved"] = {k: first[k] for k in ("diff", "ci95", "a_better", "b_better")}
    return out


def label_sheet(results: dict, questions: dict[str, Question], n: int, seed: int = 0) -> list[dict]:
    """A random sample of judged answers, without the judge's verdict, for a person to label.

    An answer that two settings gave in identical words appears once, since the judge saw it once.
    """
    judged, seen = [], set()
    for r in results["rows"]:
        key = (r["qid"], r["answer"].strip())
        if r["verdict"] in ("correct", "incorrect") and key not in seen:
            seen.add(key)
            judged.append(r)
    sample = random.Random(seed).sample(judged, min(n, len(judged)))
    return [{"qid": r["qid"], "setting": r["setting"], "question": questions[r["qid"]].text,
             "reference": questions[r["qid"]].answer, "answer": r["answer"], "your_verdict": ""}
            for r in sample]


def _po_kappa(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pc = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n)
             for c in ("correct", "incorrect"))
    return po, ((po - pc) / (1 - pc) if pc < 1 else (1.0 if po == 1 else float("nan")))


def agreement(results: dict, labels: list[dict], n_boot: int = 10_000, seed: int = 0) -> dict:
    """The judge against a person's labels: agreement, Cohen's kappa, and 95% intervals that
    resample questions, since one question can carry several labeled answers."""
    judge = {(r["qid"], r["setting"]): r["verdict"] for r in results["rows"]}
    labeled = [(lab["qid"], lab["your_verdict"].strip().lower(), judge[(lab["qid"], lab["setting"])])
               for lab in labels if lab["your_verdict"].strip()]
    if not labeled:
        return {"labeled": 0}
    pairs = [(a, b) for _, a, b in labeled]
    po, kappa = _po_kappa(pairs)
    by_q: dict[str, list[tuple[str, str]]] = {}
    for qid, a, b in labeled:
        by_q.setdefault(qid, []).append((a, b))
    groups = list(by_q.values())
    rng = np.random.default_rng(seed)
    pos, kappas = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        p, k = _po_kappa([x for i in pick for x in groups[i]])
        pos.append(p)
        kappas.append(k)
    kappas_ok = [k for k in kappas if not math.isnan(k)]
    return {"labeled": len(pairs), "questions": len(groups), "agree": sum(a == b for a, b in pairs),
            "agreement": po, "agreement_ci95": [float(x) for x in np.percentile(pos, [2.5, 97.5])],
            "cohens_kappa": kappa,
            "kappa_ci95": [float(x) for x in np.percentile(kappas_ok, [2.5, 97.5])] if kappas_ok else None,
            "judge_says_correct_you_dont": sum(a == "incorrect" and b == "correct" for a, b in pairs),
            "judge_says_incorrect_you_dont": sum(a == "correct" and b == "incorrect" for a, b in pairs)}
