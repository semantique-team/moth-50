import json

import pytest
from eta_breakdown import (
    KINDS,
    classify,
    compare,
    flags,
    main,
    reference_counts,
    summarize,
)


@pytest.mark.parametrize(
    ("term", "got", "kind"),
    [
        ("product", None, "deleted"),
        ("product", "", "deleted"),
        ("l'opex", "lopex", "apostrophe"),
        ("bucket", "storagebucket", "glued"),
        ("l'sdk", "dell'sdk", "glued"),
        ("path", "pat", "truncated"),
        ("cig", "chi", "substituted"),
        ("too", "to", "truncated"),
        ("favourite", "feverage", "substituted"),
    ],
)
def test_classify(term, got, kind):
    assert classify(term, got) == kind


def test_flags():
    assert flags("l'sdk") == ["elided", "no_vowel"]
    assert flags("mcv") == ["no_vowel"]
    assert flags("let's") == ["elided"]
    assert flags("cig") == []
    assert flags("it") == ["function_word"]
    assert flags("meeting") == []
    assert "no_vowel" not in flags("4")


def _errors():
    return [
        {"id": "a", "term": "l'sdk", "got": "dk"},
        {"id": "a", "term": "too", "got": "to"},
        {"id": "b", "term": "product", "got": None},
        {"id": "b", "term": "bucket", "got": "storagebucket"},
    ]


def test_summarize_counts_every_miss_once():
    summary = summarize(_errors(), {"a": "acronym", "b": "phrase"})
    assert summary["misses"] == 4
    assert sum(summary["kinds"].values()) == 4
    assert set(summary["kinds"]) == set(KINDS)
    assert summary["kinds"]["deleted"] == 1
    assert summary["kinds"]["glued"] == 1
    assert summary["flags"] == {"elided": 1, "no_vowel": 1, "function_word": 1}
    assert summary["by_category"] == {
        "acronym": {"substituted": 1, "truncated": 1},
        "phrase": {"deleted": 1, "glued": 1},
    }
    assert summarize(_errors())["by_category"] is None


def test_compare_is_a_multiset_difference_on_id_and_term():
    other = [
        {"id": "a", "term": "too", "got": "two"},
        {"id": "c", "term": "path", "got": "pat"},
    ]
    diff = compare(_errors(), other)
    assert diff["misses"] == 4
    assert diff["other_misses"] == 2
    assert diff["only_here"] == 3
    assert diff["only_other"] == 1
    assert ["l'sdk", "dk"] in diff["only_here_examples"]


def test_reference_counts_use_the_scored_words():
    rows = [
        {"norm": "apri l'sdk e il crm", "en_spans": [[1, 2], [4, 5]], "category": "acronym"},
        {"norm": "it makes sense", "en_spans": [[0, 3]], "lang": "en"},
        {"norm": "nessun termine", "en_spans": []},
    ]
    counts = reference_counts(rows)
    assert counts["english_words"] == 5
    assert counts["flags"] == {"elided": 1, "no_vowel": 2, "function_word": 1}
    assert counts["categories"] == {"acronym": 2, "en": 3}


def test_cli_writes_a_comparison(tmp_path):
    def readout(errors, eta):
        return {"metrics": {"eta_errors": errors, "eta": eta, "wer": 0.1, "category": {}}}

    ours = tmp_path / "ours.json"
    theirs = tmp_path / "theirs.json"
    ours.write_text(json.dumps(readout(_errors(), 0.8)), encoding="utf-8")
    theirs.write_text(json.dumps(readout(_errors()[:1], 0.9)), encoding="utf-8")
    out = tmp_path / "out" / "breakdown.json"
    assert main([str(ours), "--compare", str(theirs), "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["summary"]["misses"] == 4
    assert payload["compare"]["diff"]["only_here"] == 3
    with pytest.raises(SystemExit, match="refusing"):
        main([str(ours), "--out", str(out)])
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"macro_wer": 0.2}), encoding="utf-8")
    with pytest.raises(SystemExit, match="mixed"):
        main([str(bad)])
