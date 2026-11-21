from collections import Counter

from lm_corpus_overlap import analyse_set, term_tuples


def test_term_tuples_uses_en_spans_and_normalizer():
    row = {"text": "la free trial scade domani", "en_spans": [[1, 3]]}
    assert term_tuples(row) == [("free", "trial")]


def test_term_tuples_handles_elisions_and_empty_spans():
    row = {"text": "l'SDK è pronto", "en_spans": [[0, 1]]}
    assert term_tuples(row) == [("l'sdk",)]
    assert term_tuples({"text": "nessun termine", "en_spans": []}) == []


def test_analyse_set_reports_exact_four_grams_and_term_counts():
    rows = [
        {"id": "a", "text": "la free trial scade", "en_spans": [[1, 3]]},
        {"id": "b", "text": "il file è su", "en_spans": [[1, 2]]},
    ]
    report = analyse_set(
        rows,
        exact={"la free trial scade"},
        four_grams={("la", "free", "trial", "scade")},
        term_counts=Counter({("free", "trial"): 1, ("file",): 2}),
    )

    assert report["rows"] == 2
    assert report["exact_matches"] == 1
    assert report["exact_match_ids"] == ["a"]
    assert report["four_grams"]["unique"] == 2
    assert report["four_grams"]["unique_present"] == 1
    assert report["four_grams"]["unique_share"] == 0.5
    assert report["four_grams"]["occurrence_share"] == 0.5
    assert report["terms"]["multiword"]["one"] == 1
    assert report["terms"]["multiword"]["one_share"] == 1.0
    assert report["terms"]["all"]["two_plus"] == 1
    assert report["terms"]["one_examples"] == ["free trial"]
