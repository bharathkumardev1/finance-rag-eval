"""Generate the tiny synthetic corpus that tests and CI run on.

Every company, filing and figure here is invented. The corpus exists so the whole pipeline
(filter, BM25, statement routing, metrics, service, quality gate) runs in CI in seconds
without downloading anything. It has the same shapes that matter in the real filings:
narrative pages, statement pages that are mostly numbers, an exhibit page that repeats the
company name, and a quarterly filing. README numbers come from real FinanceBench data.

Run ``python tests/fixtures/make_mini.py`` to regenerate ``tests/fixtures/mini``.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent / "mini"

COMPANIES = {"Acme Corporation": "ACME", "Globex": "GLOBEX", "Initech": "INITECH"}

FACTS = {
    ("ACME", 2021): ["Acme recorded a goodwill impairment of $45 million in its aerospace unit.",
                     "Acme repurchased 3.1 million shares under its buyback program.",
                     "Acme opened a distribution center in Ohio to shorten delivery times."],
    ("ACME", 2022): ["Acme settled the coastal litigation for $80 million.",
                     "Acme named a new chief executive officer in March.",
                     "Acme exited its consumer batteries business."],
    ("GLOBEX", 2021): ["Globex issued $500 million of green bonds due 2031.",
                       "Globex wildfire mitigation spending totaled $610 million.",
                       "Globex customer count rose to 1.8 million accounts."],
    ("GLOBEX", 2022): ["Globex wildfire mitigation spending totaled $705 million.",
                       "Globex retired its last coal unit in December.",
                       "Globex raised its dividend after the general rate case."],
    ("INITECH", 2021): ["Initech acquired a data labeling startup for $95 million.",
                        "Initech headcount ended the year at 6,400 employees.",
                        "Initech moved its headquarters to Austin."],
    ("INITECH", 2022): ["Initech recorded restructuring charges of $28 million.",
                        "Initech headcount ended the year at 6,100 employees.",
                        "Initech launched a usage-based pricing tier."],
}

FILLER = ("The company operates under the regulatory and competitive conditions described in "
          "this report. Forward-looking statements involve risks and uncertainties. ") * 5


def statement(company: str, title: str, year: int, lines: list[tuple[str, int]]) -> str:
    rows = " ".join(f"{label} {v:,} {int(v * 0.93):,} {int(v * 0.88):,}" for label, v in lines)
    return f"{company} {title} (In millions) {year} {year - 1} {year - 2} {rows}"


def filing(company: str, ticker: str, year: int) -> list[str]:
    seed = sum(map(ord, ticker)) + year
    cover = (f"UNITED STATES SECURITIES AND EXCHANGE COMMISSION Washington, D.C. FORM 10-K "
             f"Annual report for the fiscal year ended December 31, {year}. {company}.")
    narrative = [f"{fact} {FILLER}" for fact in FACTS[(ticker, year)]]
    balance = statement(company, "CONSOLIDATED BALANCE SHEETS", year, [
        ("Cash and cash equivalents", 400 + seed % 97), ("Accounts receivable, net", 310),
        ("Inventories", 220), ("Total current assets", 930), ("Property, plant and equipment, net",
        1200 + seed % 50), ("Total assets", 2900), ("Accounts payable", 180),
        ("Total current liabilities", 610), ("Long-term debt", 900), ("Total equity", 1390)])
    income = statement(company, "CONSOLIDATED STATEMENTS OF OPERATIONS", year, [
        ("Net sales", 2300 + seed % 300), ("Cost of sales", 1500), ("Gross profit", 800 + seed % 300),
        ("Selling, general and administrative", 350), ("Operating income", 450),
        ("Interest expense", 40), ("Income before taxes", 410), ("Provision for income taxes", 90),
        ("Net income", 320)])
    cash = statement(company, "CONSOLIDATED STATEMENTS OF CASH FLOWS", year, [
        ("Net income", 320), ("Depreciation and amortization", 140),
        ("Net cash provided by operating activities", 520), ("Purchases of property and equipment",
        120 + seed % 60), ("Net cash used in investing activities", 260),
        ("Dividends paid", 90), ("Net cash used in financing activities", 180)])
    exhibit = (f"Exhibit 10.4 {company} deferred compensation plan. {company} and its "
               f"subsidiaries, {company} officers and {company} directors. " * 4)
    return [cover, *narrative, balance, income, cash, exhibit]


def quarterly(company: str, ticker: str, year: int, quarter: int) -> list[str]:
    cover = (f"UNITED STATES SECURITIES AND EXCHANGE COMMISSION FORM 10-Q Quarterly report for "
             f"the period ended June 30, {year}. {company}.")
    segment = (f"Segment results for the three months ended June 30, {year}: Industrial net "
               f"revenue $ 610 Safety net revenue $ 455 Aerospace net revenue $ 132 " + FILLER)
    return [cover, segment, statement(company, "CONDENSED CONSOLIDATED BALANCE SHEETS", year, [
        ("Cash and cash equivalents", 377), ("Total current assets", 905),
        ("Total current liabilities", 640), ("Total assets", 2950)])]


QUESTIONS = [
    ("q01", "How much did Acme pay to settle the coastal litigation in 2022?", "ACME_2022_10K", 1),
    ("q02", "What was Globex wildfire mitigation spending in FY2022?", "GLOBEX_2022_10K", 1),
    ("q03", "How large was the Globex green bond issuance in 2021?", "GLOBEX_2021_10K", 1),
    ("q04", "What restructuring charges did Initech record in 2022?", "INITECH_2022_10K", 1),
    ("q05", "How many shares did Acme repurchase in fiscal 2021?", "ACME_2021_10K", 2),
    ("q06", "Which startup did Initech acquire in 2021 and for how much?", "INITECH_2021_10K", 1),
    ("q07", "What is Acme's FY2021 working capital? Use the balance sheet.", "ACME_2021_10K", 4),
    ("q08", "What was Globex's FY2022 capital expenditure according to the cash flow statement?",
     "GLOBEX_2022_10K", 6),
    ("q09", "What is Initech's FY2021 gross margin based on the income statement?",
     "INITECH_2021_10K", 5),
    ("q10", "What was Acme's net revenue by segment in 2022 Q2?", "ACME_2022Q2_10Q", 1),
    ("q11", "What is the quick ratio of Globex for FY2021?", "GLOBEX_2021_10K", 4),
    ("q12", "Which company retired its last coal unit in December?", "GLOBEX_2022_10K", 2),
]


def main() -> None:
    (HERE / "pages").mkdir(parents=True, exist_ok=True)
    docs = []
    for company, ticker in COMPANIES.items():
        for year in (2021, 2022):
            name = f"{ticker}_{year}_10K"
            (HERE / "pages" / f"{name}.json").write_text(
                json.dumps(filing(company, ticker, year)), encoding="utf8")
            docs.append({"doc_name": name, "company": company, "gics_sector": "",
                         "doc_type": "10k", "doc_period": year, "doc_link": ""})
    (HERE / "pages" / "ACME_2022Q2_10Q.json").write_text(
        json.dumps(quarterly("Acme Corporation", "ACME", 2022, 2)), encoding="utf8")
    docs.append({"doc_name": "ACME_2022Q2_10Q", "company": "Acme Corporation", "gics_sector": "",
                 "doc_type": "10q", "doc_period": 2022, "doc_link": ""})
    with open(HERE / "financebench_document_information.jsonl", "w", encoding="utf8") as fh:
        for d in docs:
            fh.write(json.dumps(d) + "\n")
    company_of = {t: c for c, t in COMPANIES.items()}
    types = ("domain-relevant", "metrics-generated", "novel-generated")
    with open(HERE / "financebench_open_source.jsonl", "w", encoding="utf8") as fh:
        for i, (qid, text, doc, page) in enumerate(QUESTIONS):
            fh.write(json.dumps({
                "financebench_id": qid, "question": text, "answer": "",
                "company": company_of[doc.split("_")[0]], "doc_name": doc,
                "question_type": types[i % 3],
                "evidence": [{"evidence_text": "", "doc_name": doc, "evidence_page_num": page,
                              "evidence_text_full_page": ""}],
            }) + "\n")


if __name__ == "__main__":
    main()
