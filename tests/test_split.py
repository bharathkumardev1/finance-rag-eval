from collections import Counter

from finrag.data import load_questions
from finrag.split import load_split, make_split

PER_TYPE = {"domain-relevant": 1, "metrics-generated": 1, "novel-generated": 1}


def test_split_is_stratified_and_ignores_input_order(mini_dir):
    qs = load_questions(mini_dir)
    a = make_split(qs, PER_TYPE)
    assert a == make_split(list(reversed(qs)), PER_TYPE)
    types = {q.qid: q.question_type for q in qs}
    assert Counter(types[i] for i in a["dev"]) == Counter(PER_TYPE)
    assert len(a["test"]) == 9 and not set(a["dev"]) & set(a["test"])


def test_fixture_split_file_is_current(mini_dir):
    assert load_split(mini_dir / "split.json") == make_split(load_questions(mini_dir), PER_TYPE)
