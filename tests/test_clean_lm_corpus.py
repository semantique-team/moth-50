import json

from clean_lm_corpus import clean_lines


def _corpus(*rows):
    return ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()


def test_clean_lines_drops_normalized_exact_matches_and_keeps_original_bytes():
    corpus = _corpus(
        {"text": "La free trial scade domani.", "lang": "it"},
        {"text": "il file è su", "lang": "it"},
    )
    kept, dropped = clean_lines(corpus, {"la free trial scade domani"})

    assert dropped == ["La free trial scade domani."]
    assert len(kept) == 1
    assert json.loads(kept[0])["text"] == "il file è su"


def test_clean_lines_treats_mix_as_italian_and_keeps_unmatched_rows():
    corpus = _corpus({"text": "il 15:30 è tardi", "lang": "mix"})
    kept, dropped = clean_lines(corpus, {"il quindici trenta è tardi"})

    assert dropped == ["il 15:30 è tardi"]
    assert kept == []
