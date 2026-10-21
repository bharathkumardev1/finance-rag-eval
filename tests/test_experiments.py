from finrag import experiments as ex
from finrag.data import load_questions
from finrag.evaluate import run
from finrag.search import FULL, Options, Searcher
from finrag.split import load_split


def test_every_configuration_runs_once_and_dense_never_drops_the_name():
    configs = ex.configs(("bm25", "dense", "hybrid"))
    assert len(configs) == len(set(configs))
    assert ("bm25", FULL) in configs and ("hybrid", FULL) in configs
    assert not any(o.drop_company_terms for m, o in configs if m == "dense")
    assert ("bm25", ex.NO_ROUTE) in configs


def test_the_ladder_adds_one_piece_at_a_time():
    def pieces(o: Options) -> int:
        return (o.filter != "none") + (o.filter == "company_year") + o.quarters + o.route_statements \
            + o.drop_company_terms

    assert [pieces(o) for o in ex.LADDER] == [0, 1, 2, 3, 4, 5]


def test_headline_is_picked_on_dev_only():
    table = [
        {"config": "bm25/" + FULL.key(), "split": "dev", "hit@10": 0.5, "mrr@10": 0.3},
        {"config": "hybrid/" + FULL.key(), "split": "dev", "hit@10": 0.6, "mrr@10": 0.2},
        {"config": "bm25/" + FULL.key(), "split": "test", "hit@10": 0.9, "mrr@10": 0.9},
        {"config": "hybrid/" + FULL.key(), "split": "test", "hit@10": 0.1, "mrr@10": 0.1},
    ]
    assert ex.pick_headline({"table": table}, ("bm25", "hybrid")) == "hybrid/" + FULL.key()


def test_full_run_on_the_fixture(mini_dir):
    s = Searcher.build(mini_dir)
    questions = load_questions(mini_dir)
    split = load_split(mini_dir / "split.json")
    res = run(ex.systems(s, ("bm25",)), questions, split)
    headline = ex.pick_headline(res, ("bm25",))
    assert headline == "bm25/" + FULL.key()
    pairs = {(c["a"], c["b"]) for c in ex.comparisons(res, ("bm25",), headline)}
    assert ("bm25/company_year+quarter", "bm25/all-pages") in pairs
    report = ex.filter_report(s, questions)
    assert report["n"] == 12 and report["gold_filing_kept"] <= report["filtered"] <= 12
