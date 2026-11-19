import json

import numpy as np
import pytest
from probe_vad import TOLERANCE, collect, compare, verdict


def test_compare_counts_gaps_flips_and_gates():
    reference = np.array([0.1, 0.6, 0.9, 0.2], dtype=np.float32)
    candidate = np.array([0.1, 0.4, 0.9, 0.2], dtype=np.float32)
    result = compare(reference, candidate, window_threshold=0.5, gate=0.10)
    assert result["windows"] == 4
    assert result["max_abs"] == pytest.approx(0.2)
    assert result["window_flips"] == 1
    assert result["gate_reference"] is True
    assert result["gate_candidate"] is True


def test_verdict_needs_identical_gates_segments_and_probabilities():
    same = {"id": "a", "max_abs": 0.0, "window_flips": 0, "gate_reference": True}
    same |= {"gate_candidate": True, "segments_equal": True}
    assert verdict([same])["drop_in"] is True
    assert verdict([same | {"max_abs": TOLERANCE * 10}])["drop_in"] is False
    gate_flip = verdict([same | {"gate_candidate": False}])
    assert gate_flip["gate_disagreements"] == ["a"]
    assert gate_flip["drop_in"] is False
    assert verdict([same | {"segments_equal": False}])["segment_disagreements"] == ["a"]


def test_collect_reads_manifests_and_audio_folders(tmp_path):
    manifest = tmp_path / "noise.jsonl"
    manifest.write_text(
        json.dumps({"id": "n-1", "audio": "noise/n-1.wav"}) + "\n\n", encoding="utf-8"
    )
    folder = tmp_path / "audio"
    folder.mkdir()
    (folder / "b.wav").write_bytes(b"")
    (folder / "a.flac").write_bytes(b"")
    (folder / "notes.txt").write_text("x", encoding="utf-8")
    clips = collect([manifest], [tmp_path / "root"], [folder])
    assert clips == [
        ("noise:n-1", tmp_path / "root" / "noise/n-1.wav"),
        ("file:a.flac", folder / "a.flac"),
        ("file:b.wav", folder / "b.wav"),
    ]
