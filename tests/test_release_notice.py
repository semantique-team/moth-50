import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from release_notice import TRAINING_SOURCES, compose_notice, moth_dataset_lines

LOCK = json.loads(
    (Path(__file__).resolve().parents[1] / "data" / "sources.lock.json").read_text(encoding="utf-8")
)


def _notice(**overrides) -> str:
    arguments = {
        "whisper_mit": "WHISPER MIT TEXT",
        "silero_mit": "SILERO MIT TEXT",
        "lock": LOCK,
        "moth_lines": ["- Moth source — CC0 — https://example.org"],
        "encoder_preset": "base",
        "head_blocks": 6,
        "lm_sha256": "3be1177b" + "0" * 56,
        "vad_description": "the 8 kHz + 16 kHz graph",
    }
    arguments.update(overrides)
    return compose_notice(**arguments)


def test_notice_names_the_encoder_actually_used():
    notice = _notice()
    assert notice.startswith("MOTH-50")
    assert "``openai/whisper-base``" in notice
    assert "whisper-tiny" not in notice
    assert "e37978b90ca9030d5170a5c07aadb050351a65bb" in notice
    assert "6 additional transformer blocks and a CTC head" in notice
    assert "1 additional transformer block and" in _notice(head_blocks=1)
    assert "WHISPER MIT TEXT" in notice and "SILERO MIT TEXT" in notice


def test_notice_lists_the_lm_vad_and_every_training_source():
    notice = _notice(vad_description="16 kHz-only graph")
    assert "3be1177b" in notice
    assert "16 kHz-only graph" in notice
    assert "commit bfdc019" in notice
    assert "- Moth source — CC0 — https://example.org" in notice
    sources = {source["id"]: source for source in LOCK["sources"]}
    for key in TRAINING_SOURCES:
        assert f"- {key} " in notice
        assert sources[key]["license"] in notice
    assert "cs-fleurs" not in notice


def test_notice_requires_the_training_sources_in_the_lock():
    lock = {"sources": [source for source in LOCK["sources"] if source["id"] != "musan"]}
    with pytest.raises(KeyError, match="musan"):
        _notice(lock=lock)


def test_moth_lines_match_moth_build_notice():
    sources = [
        SimpleNamespace(citation="A (CC0)", name="a", license="CC0", url="u1"),
        SimpleNamespace(citation="A (CC0)", name="a", license="CC0", url="u1"),
        SimpleNamespace(citation=None, name="b", license="CC-BY 4.0", url="u2"),
        SimpleNamespace(citation=None, name="c", license="verify later", url="u3"),
    ]
    assert moth_dataset_lines(sources) == ["- A (CC0) — CC0 — u1", "- b — CC-BY 4.0 — u2"]
