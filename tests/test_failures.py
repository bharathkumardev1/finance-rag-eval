from finrag.data import Evidence, Question, load_questions
from finrag.evaluate import run
from finrag.experiments import systems
from finrag.failures import CATEGORIES, classify, taxonomy
from finrag.search import FULL, Options, Searcher
from finrag.split import load_split


def question(text, doc, page, company="Acme Corporation"):
    return Question("x", text, "", company, doc, "t", (Evidence(doc, page),))


def test_each_stage_that_can_lose_the_page(mini_dir):
    s = Searcher.build(mini_dir)
    no_company = question("Which company opened a distribution center in Ohio?", "ACME_2021_10K", 3)
    assert classify(s, no_company, [], FULL) == "no_company"
    wrong_year = question("What did Acme repurchase in 2022?", "ACME_2021_10K", 2)
    assert classify(s, wrong_year, [], FULL) == "filter_dropped_filing"
    unrouted = question("What was Acme's FY2021 cash position?", "ACME_2021_10K", 4)
    assert classify(s, unrouted, [], FULL) == "statement_not_detected"
    routed = question("What was Acme's FY2021 working capital?", "ACME_2021_10K", 4)
    assert classify(s, routed, [], FULL) == "statement_outranked"
    vocab = question("Who leads Acme now, as of 2022?", "ACME_2022_10K", 2)
    assert classify(s, vocab, [], FULL) == "vocabulary_gap"
    words = question("Acme 2022 chief executive officer named March", "ACME_2022_10K", 2)
    assert classify(s, words, [], FULL) == "outranked"
    no_year = question("How much did Acme pay in the coastal litigation?", "ACME_2022_10K", 1)
    other_rows = [s.corpus.offsets["ACME_2021_10K"][0] + i for i in range(8)]
    assert classify(s, no_year, other_rows, Options("company", route_statements=True)) == (
        "other_filing_won")


def test_taxonomy_counts_only_misses(mini_dir):
    s = Searcher.build(mini_dir)
    questions = load_questions(mini_dir)
    res = run(systems(s, ("bm25",)), questions, load_split(mini_dir / "split.json"))
    config = "bm25/" + FULL.key()
    tax = taxonomy(s, questions, res, config, FULL, None)
    hits = sum(any(r <= 10 for r in x["runs"][config]["gold_ranks"]) for x in res["questions"].values())
    assert tax["misses"] == len(questions) - hits == sum(tax["counts"].values())
    assert list(tax["counts"]) == list(CATEGORIES)
