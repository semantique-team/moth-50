import random

import pytest
from keyterm_probe import select_terms

CANDIDATES = [f"term{i}" for i in range(50)] + ["deploy", "meeting"]


def _select(context, oracle=("deploy", "meeting"), words=None, **kwargs):
    words = words or ["ho", "fatto", "il", "deploy", "del", "meeting"]
    options = {"recall": 1.0, "distractors": 10, "seed": 0, "clip_id": "R-S1-CAL-0001"}
    options.update(kwargs)
    return select_terms(list(oracle), words, CANDIDATES, context=context, **options)


def test_none_selects_nothing():
    assert _select("none") == ([], [])


def test_oracle_keeps_every_term_at_full_recall():
    assert _select("oracle") == (["deploy", "meeting"], [])


def test_oracle_distractors_match_the_e022_draw():
    kept, chosen = _select("oracle-distractors")
    pool = [term for term in CANDIDATES if term not in {"deploy", "meeting"}]

    assert kept == ["deploy", "meeting"]
    assert chosen == random.Random("0:R-S1-CAL-0001").sample(pool, 10)


def test_partial_recall_is_seeded_and_leaves_distractors_unchanged():
    oracle = [f"term{i}" for i in range(40)]
    first = _select("oracle-distractors", oracle=oracle, recall=0.5)
    second = _select("oracle-distractors", oracle=oracle, recall=0.5)
    full = _select("oracle-distractors", oracle=oracle)

    assert first == second
    assert set(first[0]) < set(oracle)
    assert 0 < len(first[0]) < len(oracle)
    assert first[1] == full[1]


def test_zero_recall_keeps_no_oracle_term():
    assert _select("oracle", recall=0.0) == ([], [])


def test_distractors_exclude_every_reference_word():
    words = ["term1", "term2", "term3", "e", "deploy"]
    kept, chosen = _select("distractors", oracle=["deploy"], words=words, distractors=40)

    assert kept == []
    assert len(chosen) == 40
    assert not set(chosen) & set(words)


def test_too_small_pool_is_an_error():
    with pytest.raises(ValueError, match="distractors available"):
        _select("distractors", distractors=len(CANDIDATES))


def test_unknown_context_is_an_error():
    with pytest.raises(ValueError, match="unknown context"):
        _select("glossary")
