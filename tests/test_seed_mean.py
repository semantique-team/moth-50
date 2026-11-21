from seed_mean import mean_system, metric


def _utterance(ref=10, sub=0, dele=0, ins=0, en_ref=4, en_hit=4):
    return {
        "ref_words": ref,
        "sub": sub,
        "del": dele,
        "ins": ins,
        "en_ref": en_ref,
        "en_hit": en_hit,
    }


def test_metric_pooled_wer_and_eta():
    per_utterance = {
        "a": _utterance(ref=10, sub=1, en_ref=4, en_hit=3),
        "b": _utterance(ref=10, dele=1, en_ref=4, en_hit=4),
    }
    result = metric(per_utterance, {"all": ["a", "b"]})

    assert result["wer"] == 0.1
    assert result["eta"] == 7 / 8


def test_metric_macro_averages_languages_equally():
    per_utterance = {
        "it1": _utterance(ref=10, sub=2, en_ref=0, en_hit=0),
        "it2": _utterance(ref=10, en_ref=0, en_hit=0),
        "en1": _utterance(ref=100, en_ref=4, en_hit=2),
    }
    result = metric(per_utterance, {"it": ["it1", "it2"], "en": ["en1"]})

    assert result["wer"] == (0.1 + 0.0) / 2
    assert result["eta"] == 0.5


def test_mean_system_averages_counts_across_seeds():
    first = {"u": _utterance(sub=0, en_hit=2)}
    second = {"u": _utterance(sub=2, en_hit=4)}
    mean = mean_system([first, second], ["u"])

    assert mean["u"]["sub"] == 1.0
    assert mean["u"]["en_hit"] == 3.0
    assert mean["u"]["ref_words"] == 10
