import json

import numpy as np
import pytest
from e027_audit import sample_audit, score_audit, stratum
from e027_share import manifest_hours, tts_repeat
from e027_tts_render import (
    VOICES,
    load_pins,
    manifest_row,
    peak_normalize,
    qc_reason,
    renders,
    runs,
)
from e027_tts_script import (
    Lexicon,
    keep_reason,
    row_lang,
    script_id,
    segments,
    split_elision,
)

LEXICON = Lexicon(
    terms=frozenset({"sdk", "deploy", "file"}),
    english=frozenset({"meeting", "server", "come", "password"}),
    italian=frozenset({"come", "server", "file", "il", "ho"}),
)


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace(",", " ").replace(".", " ").split())


def test_english_needs_a_term_or_an_english_word_that_is_not_italian():
    assert LEXICON.is_english("sdk")
    assert LEXICON.is_english("file")  # a listed term beats the Italian veto
    assert LEXICON.is_english("meeting")
    assert LEXICON.is_english("the")
    assert not LEXICON.is_english("come")  # Italian homograph
    assert not LEXICON.is_english("server")  # frequent in the Italian training text
    assert not LEXICON.is_english("domani")


def test_elision_is_split_only_before_an_english_word():
    assert split_elision("l'SDK,") == ("l'", "SDK,")
    assert split_elision("dell’API") == ("dell'", "API")
    assert split_elision("deploy") is None
    pieces = segments("Ho aggiornato l'SDK e l'azienda, meeting.", "acronym", LEXICON, _normalize)
    assert pieces == [
        ["Ho", "it"],
        ["aggiornato", "it"],
        ["l'", "it"],
        ["SDK", "en"],
        ["e", "it"],
        ["l'azienda,", "it"],
        ["meeting.", "en"],
    ]


def test_control_categories_are_one_language():
    english = segments("Il server come va", "control_en", LEXICON, _normalize)
    italian = segments("The deploy is done", "control_it", LEXICON, _normalize)
    assert {tag for _, tag in english} == {"en"}
    assert {tag for _, tag in italian} == {"it"}
    assert row_lang("control_en") == "en"
    assert row_lang("control_it") == "it"
    assert row_lang("switch") == "mix"


def test_keep_reason_drops_digits_length_eval_overlap_and_duplicates():
    grams = {("ho", "fatto", "il", "deploy")}
    clean = "ieri ho fatto un deploy"
    assert keep_reason("Ho 3 file da aprire", "ho tre file da aprire", set(), set()) == "digits"
    assert keep_reason("Ciao a tutti", "ciao a tutti", set(), set()) == "length"
    assert keep_reason("x", "ieri ho fatto il deploy", grams, set()) == "eval_4gram"
    assert keep_reason("x", clean, grams, {clean}) == "duplicate"
    assert keep_reason("x", clean, grams, set()) is None
    assert script_id("E025-00042") == "E027-00042"


def test_runs_join_same_language_pieces():
    pieces = [["Ho", "it"], ["fatto", "it"], ["il", "it"], ["deploy", "en"], ["ieri.", "it"]]
    assert runs(pieces, "it") == [("Ho fatto il deploy ieri.", "it")]
    assert runs(pieces, "mix") == [("Ho fatto il", "it"), ("deploy", "en"), ("ieri.", "it")]
    with pytest.raises(ValueError, match="mode"):
        runs(pieces, "en")


def test_mix_renders_need_english_and_kokoro():
    english = {"segments": [["il", "it"], ["deploy", "en"]]}
    italian = {"segments": [["il", "it"], ["treno", "it"]]}
    assert renders(english, VOICES["sara"]) == ["it", "mix"]
    assert renders(italian, VOICES["sara"]) == ["it"]
    assert renders(english, VOICES["paola"]) == ["it"]


def test_every_voice_file_is_pinned_in_the_lock():
    pins = load_pins()
    for voice in VOICES.values():
        for name in voice.files:
            assert len(pins[name]) == 64


def test_peak_normalize_and_qc():
    audio = peak_normalize(np.array([0.0, 0.5, -0.25]))
    assert np.isclose(np.max(np.abs(audio)), 0.7)
    assert not np.any(peak_normalize(np.zeros(3)))
    assert qc_reason(0.5, 2) == "too_short"
    assert qc_reason(31.0, 40) == "too_long"
    assert qc_reason(10.0, 5) == "too_slow"
    assert qc_reason(1.0, 7) == "too_fast"
    assert qc_reason(3.0, 8) is None


def test_manifest_row_keeps_category_on_mixed_rows_only():
    row = {"id": "E027-00001", "text": "Il deploy.", "norm": "il deploy", "category": "switch"}
    sara, paola = VOICES["sara"], VOICES["paola"]
    mixed = manifest_row(row | {"lang": "mix"}, "sara", sara, "mix", "a.wav", 1.5, "x")
    english = manifest_row(row | {"lang": "en"}, "paola", paola, "it", "b.wav", 1.5, "y")
    assert mixed["id"] == "E027-00001-sara-mix"
    assert mixed["category"] == "switch"
    assert mixed["split"] == "train"
    assert english["category"] is None
    assert stratum(mixed) == "sara-mix"


def _clip(index: int) -> dict:
    return {"id": f"c{index:02d}", "speaker": "tts-sara", "tts": {"mode": "it"}, "text": "t"}


def test_audit_sample_is_seeded_and_the_drop_rule_is_two_fails():
    rows = [_clip(index) | {"audio": "a"} for index in range(30)]
    first = sample_audit(rows, 10, 0)
    assert first == sample_audit(rows, 10, 0)
    assert len(first) == 10
    keep = [dict(row, verdict="ok") for row in first]
    keep[0]["verdict"] = "fail"
    drop = [dict(row, verdict="fail" if i < 2 else "ok") for i, row in enumerate(first)]
    assert score_audit(keep)["sara-it"]["keep"] is True
    assert score_audit(drop)["sara-it"]["keep"] is False
    with pytest.raises(ValueError, match="verdict"):
        score_audit(first)


def test_share_repeat(tmp_path):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        "".join(json.dumps({"duration": 1800.0}) + "\n" for _ in range(4)), encoding="utf-8"
    )
    assert manifest_hours(manifest) == 2.0
    assert tts_repeat(595.0, 22.0) == 3
    assert tts_repeat(595.0, 200.0) == 1
    with pytest.raises(ValueError, match="share"):
        tts_repeat(595.0, 22.0, share=1.0)
