import hashlib
import json

import pytest
from moth_release import MAX_BUDGET_BYTES, validate_release

ASSETS = {
    "model.int8.onnx": b"onnx",
    "vocab.json": b"{}",
    "mel_filters.bin": b"mel",
    "silero_vad.onnx": b"vad",
    "lm_bpe3.bin": b"MOTHLMB1" + b"\0" * 8,
}


def _release(root, name="moth-50", extra_bytes=0):
    for file, payload in ASSETS.items():
        (root / file).write_bytes(payload + b"\0" * (extra_bytes if file == "lm_bpe3.bin" else 0))
    config = {
        "name": name,
        "version": "0.0.1",
        "sample_rate": 16000,
        "features": {
            "n_fft": 400,
            "hop": 160,
            "n_mels": 80,
            "window": "hann_periodic",
            "center": True,
            "pad_mode": "reflect",
            "drop_last_frame": True,
            "floor_range": 8.0,
            "scale": "(x+4)/4",
            "mel_filters": "mel_filters.bin",
        },
        "model": {
            "file": "model.int8.onnx",
            "input": "mel",
            "output": "log_probs",
            "frame_rate_hz": 50,
            "max_seconds": 30,
        },
        "tokenizer": {"vocab": "vocab.json", "blank_id": 0, "space_marker": "\u2581"},
        "decode": {
            "default": "beam",
            "beam": 12,
            "top_k": 10,
            "lm": "lm_bpe3.bin",
            "lm_weight": 0.55,
            "token_bonus": 2.7,
        },
        "vad": {"file": "silero_vad.onnx", "window": 512, "context": 64, "threshold": 0.1},
        "files": {
            file: "sha256:" + hashlib.sha256((root / file).read_bytes()).hexdigest()
            for file in ASSETS
        },
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "NOTICE").write_text("notice\n", encoding="utf-8")
    return root


def test_default_budget_is_the_moth50_ceiling():
    assert MAX_BUDGET_BYTES == 50_000_000


def test_budget_above_50mb_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="budget_bytes"):
        validate_release(_release(tmp_path), MAX_BUDGET_BYTES + 1)


@pytest.mark.parametrize("name", ["moth-50", "moth-stt"])
def test_both_package_names_validate(tmp_path, name):
    _, report = validate_release(_release(tmp_path, name=name))
    assert report["budget_bytes"] == 50_000_000
    assert report["remaining_bytes"] == 50_000_000 - report["total_bytes"]


def test_unknown_package_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="name must be"):
        validate_release(_release(tmp_path, name="other"))


def test_explicit_smaller_budget_is_enforced(tmp_path):
    root = _release(tmp_path, extra_bytes=2_000)
    with pytest.raises(ValueError, match="exceeds budget"):
        validate_release(root, 1_000)
