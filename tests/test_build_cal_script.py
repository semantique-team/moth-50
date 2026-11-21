import pytest
from build_cal_script import spans_for


def test_spans_for_finds_multiword_terms():
    assert spans_for("la free trial scade domani", ["free trial"]) == [[1, 3]]


def test_spans_for_finds_multiple_terms_without_overlap():
    assert spans_for("il file è sul disco", ["file", "disco"]) == [[1, 2], [4, 5]]


def test_spans_for_accepts_whole_english_sentence():
    text = "the path along the river is quiet"
    assert spans_for(text, [text]) == [[0, 7]]


def test_spans_for_rejects_a_term_that_is_not_there():
    with pytest.raises(ValueError, match="not found"):
        spans_for("la free trial scade", ["user experience"])
