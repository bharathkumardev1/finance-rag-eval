import pytest

from finrag.data import DocMeta
from finrag.filters import MetadataFilter, load_aliases


def meta(name, company, doc_type, *periods):
    return DocMeta(name, company, doc_type, periods, "")


METAS = {m.doc_name: m for m in [
    meta("JOHNSON_JOHNSON_2022_10K", "Johnson & Johnson", "10k", 2022),
    meta("JOHNSON_JOHNSON_2023Q2_EARNINGS", "Johnson & Johnson", "earnings", 2023),
    meta("JOHNSON_JOHNSON_2022Q4_EARNINGS", "Johnson & Johnson", "earnings", 2022),
    meta("AMERICANWATERWORKS_2022_10K", "American Water Works", "10k", 2022),
    meta("AMERICANEXPRESS_2022_10K", "American Express", "10k", 2022),
    meta("FOOTLOCKER_2023_annualreport", "Foot Locker", "10k", 2022, 2023),
]}
JNJ = {"Johnson & Johnson": ["JnJ"]}


def test_companies_years_and_quarters_come_from_the_question():
    f = MetadataFilter(METAS, JNJ)
    assert f.companies_in("What was JnJ's FY2022 revenue?") == {"Johnson & Johnson"}
    assert f.companies_in("American Water Works capex") == {"American Water Works"}
    assert f.companies_in("American Express card spend") == {"American Express"}  # longest wins
    assert f.years_in("Compare FY21 with fiscal 2022") == {2021, 2022}
    assert f.quarters_in("Q2 of FY2023 vs 2022 Q4, and the second fiscal quarter of 2021") == {
        (2023, 2), (2022, 4), (2021, 2)}
    assert f.quarters_in("Q22023 and Q2'2023") == {(2023, 2)}


def test_filter_levels_and_fallbacks():
    f = MetadataFilter(METAS, JNJ)
    jnj = {d for d in METAS if d.startswith("JOHNSON")}
    assert f.allowed_docs("JnJ margin", "company") == jnj
    assert f.allowed_docs("JnJ FY2022 margin", "company_year") == {
        "JOHNSON_JOHNSON_2022_10K", "JOHNSON_JOHNSON_2022Q4_EARNINGS"}
    assert f.allowed_docs("JnJ sales in Q2 of FY2023", "company_year") == {
        "JOHNSON_JOHNSON_2023Q2_EARNINGS"}
    assert f.allowed_docs("JnJ 2019 results", "company_year") == jnj  # no 2019 filing: back to company
    assert f.allowed_docs("Which segment grew fastest?", "company_year") is None
    assert f.allowed_docs("JnJ anything", "none") is None
    assert {"johnson", "jnj"} <= f.company_terms({"Johnson & Johnson"})
    with pytest.raises(ValueError, match="unknown filter level"):
        f.allowed_docs("JnJ", "sector")


def test_a_filing_listed_under_two_years_matches_either():
    f = MetadataFilter(METAS)
    for year in (2022, 2023):
        assert f.allowed_docs(f"Foot Locker FY{year} sales", "company_year") == {
            "FOOTLOCKER_2023_annualreport"}


def test_quarter_narrowing_can_be_switched_off():
    f = MetadataFilter(METAS, JNJ)
    assert f.allowed_docs("JnJ sales in 2022 Q4", "company_year") == {"JOHNSON_JOHNSON_2022Q4_EARNINGS"}
    assert f.allowed_docs("JnJ sales in 2022 Q4", "company_year", quarters=False) == {
        "JOHNSON_JOHNSON_2022_10K", "JOHNSON_JOHNSON_2022Q4_EARNINGS"}


def test_aliases_for_companies_without_filings_are_ignored():
    f = MetadataFilter(METAS, {"JPMorgan": ["JPM"]})
    assert f.allowed_docs("JPM net revenue in 2022", "company_year") is None


def test_aliases_file(tmp_path):
    path = tmp_path / "aliases.yaml"
    path.write_text("Johnson & Johnson: [JnJ]  # a comment\nAmerican Express: [AMEX, Amex]\n")
    assert load_aliases(path) == {"Johnson & Johnson": ["JnJ"], "American Express": ["AMEX", "Amex"]}


def test_shipped_aliases_cite_dev_questions_only():
    import re
    from pathlib import Path

    from finrag.split import load_split

    root = Path(__file__).resolve().parents[1]
    text = (root / "configs" / "aliases.yaml").read_text()
    cited = set(re.findall(r"financebench_id_\d+", text))
    dev = set(load_split(root / "configs" / "split.json")["dev"])
    assert cited and cited <= dev
    assert set(load_aliases(root / "configs" / "aliases.yaml")) == {
        "American Express", "Johnson & Johnson", "JPMorgan", "MGM Resorts"}
