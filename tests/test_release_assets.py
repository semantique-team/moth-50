import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import release_assets
from release_assets import A3_LM, ASSETS, PRE_A3_LM_SHA256, lm_label, resolve_lm, verify

LOCK = Path(__file__).resolve().parents[1] / "data" / "sources.lock.json"


def _file(tmp_path: Path, name: str, payload: bytes) -> tuple[Path, str]:
    path = tmp_path / name
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def test_pins_match_the_source_lock():
    sources = {
        source["id"]: source for source in json.loads(LOCK.read_text(encoding="utf-8"))["sources"]
    }
    lm = sources["moth-0.8.1-lm-a3"]
    assert lm["sha256"] == A3_LM.sha256
    assert lm["bytes"] == A3_LM.size
    assert lm["revision"] == release_assets.MOTH_081_REVISION
    model = sources["moth-0.8.1-model"]
    assert model["sha256"] == ASSETS["moth-0.8.1-model"].sha256
    vad = sources["silero-vad"]
    assert vad["sha256"] == ASSETS["vad"].sha256
    assert vad["revision"] == release_assets.SILERO_REVISION


def test_the_pre_a3_lm_is_not_the_a3_lm():
    assert PRE_A3_LM_SHA256 != A3_LM.sha256
    assert lm_label(A3_LM.sha256) == "a3-0.8.1"
    assert lm_label(PRE_A3_LM_SHA256) == "pre-a3-0.5.0-0.8.0"
    assert lm_label("0" * 64) == "other"


def test_verify_checks_presence_size_and_hash(tmp_path):
    path, digest = _file(tmp_path, "lm.bin", b"abc")
    asset = dataclasses.replace(A3_LM, path=path, sha256=digest, size=3)
    assert verify(asset) == path
    with pytest.raises(FileNotFoundError, match="release_assets.py fetch lm"):
        verify(dataclasses.replace(asset, path=tmp_path / "missing.bin"))
    with pytest.raises(ValueError, match="bytes"):
        verify(dataclasses.replace(asset, size=4))
    with pytest.raises(ValueError, match="sha256"):
        verify(dataclasses.replace(asset, sha256="0" * 64))


def test_resolve_lm_defaults_to_the_pinned_a3_file(tmp_path, monkeypatch):
    path, digest = _file(tmp_path, "lm_bpe3.bin", b"a3")
    monkeypatch.setattr(
        release_assets, "A3_LM", dataclasses.replace(A3_LM, path=path, sha256=digest, size=2)
    )
    resolved = release_assets.resolve_lm()
    assert resolved == {"path": path, "sha256": digest, "label": "a3-0.8.1", "is_a3": True}


def test_resolve_lm_rejects_another_lm_unless_asked(tmp_path):
    path, digest = _file(tmp_path, "other.bin", b"not a3")
    with pytest.raises(ValueError, match="allow-other-lm"):
        resolve_lm(path)
    resolved = resolve_lm(path, allow_other=True)
    assert resolved["is_a3"] is False
    assert resolved["label"] == "other"
    assert resolved["sha256"] == digest
    with pytest.raises(FileNotFoundError):
        resolve_lm(tmp_path / "missing.bin", allow_other=True)
