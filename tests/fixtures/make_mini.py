"""Generate the tiny synthetic corpus used by tests and CI.

The filings, companies, and figures are invented. They exist only so the full pipeline
(chunking, filtering, retrieval, metrics, quality gate) can run in CI in seconds without
downloading anything. Benchmark results in README.md come from the real FinanceBench data.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent / "mini"

COMPANIES = {
    "Acme Corporation": ("ACME", "Industrials"),
    "Globex": ("GLOBEX", "Utilities"),
    "Initech": ("INITECH", "Information Technology"),
}

FACTS = {
    ("ACME", 2021): ["Acme capital expenditures were $120 million in fiscal 2021.",
                     "Acme net revenue reached $2,310 million, driven by safety products.",
                     "Acme recorded a goodwill impairment of $45 million in its aerospace unit.",
                     "Acme repurchased 3.1 million shares under its buyback program."],
    ("ACME", 2022): ["Acme capital expenditures were $155 million in fiscal 2022.",
                     "Acme net revenue reached $2,505 million as pricing offset volume declines.",
                     "Acme settled the coastal litigation for $80 million.",
                     "Acme opened a new distribution center in Ohio."],
    ("GLOBEX", 2021): ["Globex rate base grew to $9.4 billion after regulatory approval.",
                       "Globex wildfire mitigation spending totaled $610 million.",
                       "Globex issued $500 million of green bonds due 2031.",
                       "Globex customer count rose to 1.8 million accounts."],
    ("GLOBEX", 2022): ["Globex rate base grew to $10.1 billion after the general rate case.",
                       "Globex wildfire mitigation spending totaled $705 million.",
                       "Globex dividend per share increased to $1.42.",
                       "Globex retired its last coal unit in December."],
    ("INITECH", 2021): ["Initech subscription revenue was $880 million, up 22 percent.",
                        "Initech research and development expense was $310 million.",
                        "Initech acquired a data labeling startup for $95 million.",
                        "Initech headcount ended the year at 6,400 employees."],
    ("INITECH", 2022): ["Initech subscription revenue was $1,040 million, up 18 percent.",
                        "Initech research and development expense was $365 million.",
                        "Initech recorded restructuring charges of $28 million.",
                        "Initech headcount ended the year at 6,100 employees."],
}

FILLER = ("The company operates under the regulatory and competitive conditions described in "
          "this report. Forward-looking statements involve risks and uncertainties. ") * 6

QUESTIONS = [
    ("q1", "What were Acme's FY2021 capital expenditures?", "ACME_2021_10K", 0),
    ("q2", "How much did Acme pay to settle the coastal litigation in 2022?", "ACME_2022_10K", 2),
    ("q3", "What was Globex wildfire mitigation spending in FY2022?", "GLOBEX_2022_10K", 1),
    ("q4", "How large was the Globex green bond issuance in 2021?", "GLOBEX_2021_10K", 2),
    ("q5", "What was Initech subscription revenue in FY2022?", "INITECH_2022_10K", 0),
    ("q6", "What restructuring charges did Initech record in 2022?", "INITECH_2022_10K", 2),
    ("q7", "How many shares did Acme repurchase in fiscal 2021?", "ACME_2021_10K", 3),
    ("q8", "What dividend per share did Globex pay in FY2022?", "GLOBEX_2022_10K", 2),
    ("q9", "Which startup did Initech acquire in 2021 and for how much?", "INITECH_2021_10K", 2),
]


def main() -> None:
    (HERE / "pages").mkdir(parents=True, exist_ok=True)
    docs = []
    for company, (ticker, sector) in COMPANIES.items():
        for year in (2021, 2022):
            name = f"{ticker}_{year}_10K"
            pages = [f"{fact} {FILLER}" for fact in FACTS[(ticker, year)]]
            (HERE / "pages" / f"{name}.json").write_text(json.dumps(pages), encoding="utf8")
            docs.append({"doc_name": name, "company": company, "gics_sector": sector,
                         "doc_type": "10k", "doc_period": year, "doc_link": ""})
    with open(HERE / "financebench_document_information.jsonl", "w", encoding="utf8") as fh:
        for d in docs:
            fh.write(json.dumps(d) + "\n")
    companies = {f"{t}": c for c, (t, _) in COMPANIES.items()}
    with open(HERE / "financebench_open_source.jsonl", "w", encoding="utf8") as fh:
        for qid, text, doc, page in QUESTIONS:
            fh.write(json.dumps({
                "financebench_id": qid, "question": text, "answer": "", "company":
                companies[doc.split("_")[0]], "doc_name": doc, "question_type": "synthetic",
                "evidence": [{"evidence_text": "", "doc_name": doc, "evidence_page_num": page,
                              "evidence_text_full_page": ""}],
            }) + "\n")


if __name__ == "__main__":
    main()
