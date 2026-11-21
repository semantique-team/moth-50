import math

import pytest
from match_audio_level import gain_for, set_levels, write_at_gain


def test_gain_for_subtracts_the_medians():
    assert gain_for({"median_dbfs": -34.0}, {"median_dbfs": -18.0}) == 16.0


def test_write_at_gain_scales_every_file_and_preserves_frames(tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    source = tmp_path / "source"
    out = tmp_path / "out"
    source.mkdir()
    for index, amplitude in ((1, 0.25), (2, 0.5)):
        sf.write(
            str(source / f"S1_{index:04d}.wav"),
            (np.ones(4800) * amplitude).astype("float32"),
            48000,
            subtype="PCM_16",
        )

    before = set_levels(source)
    result = write_at_gain(source, out, -6.0)

    assert result["files"] == 2
    assert result["clipped_samples"] == 0
    assert math.isclose(result["median_dbfs"], before["median_dbfs"] - 6.0, abs_tol=0.05)
    for name in ("S1_0001.wav", "S1_0002.wav"):
        source_info = sf.info(str(source / name))
        out_info = sf.info(str(out / name))
        assert out_info.frames == source_info.frames
        assert out_info.samplerate == source_info.samplerate
