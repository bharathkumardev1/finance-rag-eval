"""Answer generation on the test split, scored apart from retrieval.

A generator model answers each test question from the text of five pages, in two settings:

* ``retrieved``: the top 5 pages of a retrieval system, read from results/eval.json, so this
  step never reruns retrieval or needs the MiniLM embeddings.
* ``gold``: the question's gold pages, so the gap between the two settings is what retrieval
  costs and what's left in ``gold`` is the generator's own error.

A stronger judge model compares each answer with FinanceBench's reference answer and says
correct or incorrect. An answer of exactly ``NOT IN PAGES`` is scored as declined without
asking the judge. ``finrag label-judge`` samples judged answers for a person to label, and
``finrag judge-agreement`` measures how often the judge agrees with that person.

Every API call goes through a recording keyed by a hash of the request, with the model the
API reports, the tokens and the cost. A rerun replays the recording, and ``--replay`` makes
a run that needs no API key, which is how CI checks the committed results. Spending is
checked before each call against a worst-case estimate, and the run stops before it can
pass the cap. The cap covers everything in the recording, not just one run.
"""

from __future__ import annotations

import hashlib
import json
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
DECLINE = "NOT IN PAGES"
MAX_ANSWER_TOKENS = 4000
MAX_JUDGE_TOKENS = 1000
CHARS_PER_TOKEN = 3  # worst case for the pre-call estimate; English filings run about 4

ANSWER_INSTRUCTIONS = (
    "You answer questions about public company filings using only the filing pages provided. "
    "Give the figure or fact asked for, with units and the fiscal period, then one sentence "
    "saying which page it comes from and how you computed it if you did. If the pages don't "
    f"contain what you need, reply with exactly {DECLINE} and nothing else."
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
        if self.path.exists():
            for line in self.path.read_text(encoding="utf8").splitlines():
                rec = json.loads(line)
                self.calls[rec["key"]] = rec
        self.replayed = self.made = 0
        self.budget.spent = sum(rec.get("cost_usd", 0.0) for rec in self.calls.values())

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
        self.calls[k] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf8") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.made += 1
        return reply, False


def answer_request(q: Question, pages: list[tuple[str, int, str]], model: str) -> dict:
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


def page_sets(q: Question, retrieved: list[list]) -> dict[str, list[tuple[str, int]]]:
    return {"retrieved": [(d, p) for d, p in retrieved[:TOP_PAGES]],
            "gold": sorted(gold_pages(q))[:TOP_PAGES]}


def run(questions: list[Question], retrieved: dict[str, list[list]], page_text: Callable[[str, int], str],
        recorder: Recorder, generator: str = GENERATOR, judge: str = JUDGE) -> dict:
    """Answer and judge every question in both settings; stops cleanly at the budget cap."""
    rows = []
    stopped = None
    try:
        for q in questions:
            for setting, keys in page_sets(q, retrieved[q.qid]).items():
                pages = [(d, p, page_text(d, p)) for d, p in keys]
                tag = {"qid": q.qid, "setting": setting, "step": "answer"}
                ans, _ = recorder.call(answer_request(q, pages, generator), tag)
                row = {"qid": q.qid, "type": q.question_type, "setting": setting,
                       "pages": [list(k) for k in keys], "answer": ans.text, "answer_model": ans.model,
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
    except BudgetExceeded as e:
        stopped = str(e)
    return {"generator": generator, "judge": judge, "top_pages": TOP_PAGES, "rows": rows,
            "summary": summarize(rows), "stopped": stopped,
            "calls": {"made": recorder.made, "replayed": recorder.replayed},
            "recorded_spend_usd": round(recorder.budget.spent, 6), "cap_usd": recorder.budget.cap}


def summarize(rows: list[dict]) -> dict:
    out: dict = {"settings": {}}
    for setting in ("retrieved", "gold"):
        rs = [r for r in rows if r["setting"] == setting]
        if not rs:
            continue
        correct = np.array([r["verdict"] == "correct" for r in rs], dtype=float)
        secs = np.array([r["answer_seconds"] for r in rs])
        out["settings"][setting] = {
            "n": len(rs), "accuracy": float(correct.mean()), "accuracy_ci95": list(bootstrap_ci(correct)),
            "declined": sum(r["verdict"] == "declined" for r in rs),
            "incorrect": sum(r["verdict"] == "incorrect" for r in rs),
            "cost_usd": float(sum(r["cost_usd"] for r in rs)),
            "answer_p50_s": float(np.percentile(secs, 50)), "answer_p95_s": float(np.percentile(secs, 95)),
            "by_type": {t: float(np.mean([r["verdict"] == "correct" for r in rs if r["type"] == t]))
                        for t in sorted({r["type"] for r in rs})},
        }
    both = {}
    for r in rows:
        both.setdefault(r["qid"], {})[r["setting"]] = float(r["verdict"] == "correct")
    paired = [(v["gold"], v["retrieved"]) for v in both.values() if len(v) == 2]
    if paired:
        g, rt = np.array(paired).T
        out["gold_minus_retrieved"] = paired_diff(g, rt)
    return out


def label_sheet(results: dict, questions: dict[str, Question], n: int, seed: int = 0) -> list[dict]:
    """A random sample of judged answers, without the judge's verdict, for a person to label."""
    judged = [r for r in results["rows"] if r["verdict"] in ("correct", "incorrect")]
    sample = random.Random(seed).sample(judged, min(n, len(judged)))
    return [{"qid": r["qid"], "setting": r["setting"], "question": questions[r["qid"]].text,
             "reference": questions[r["qid"]].answer, "answer": r["answer"], "your_verdict": ""}
            for r in sample]


def agreement(results: dict, labels: list[dict]) -> dict:
    judge = {(r["qid"], r["setting"]): r["verdict"] for r in results["rows"]}
    pairs = [(lab["your_verdict"].strip().lower(), judge[(lab["qid"], lab["setting"])])
             for lab in labels if lab["your_verdict"].strip()]
    if not pairs:
        return {"labeled": 0}
    agree = sum(a == b for a, b in pairs)
    po = agree / len(pairs)
    pc = sum((sum(a == c for a, _ in pairs) / len(pairs)) * (sum(b == c for _, b in pairs) / len(pairs))
             for c in ("correct", "incorrect"))
    kappa = (po - pc) / (1 - pc) if pc < 1 else 1.0
    return {"labeled": len(pairs), "agree": agree, "agreement": po, "cohens_kappa": kappa,
            "judge_says_correct_you_dont": sum(a == "incorrect" and b == "correct" for a, b in pairs),
            "judge_says_incorrect_you_dont": sum(a == "correct" and b == "incorrect" for a, b in pairs)}
