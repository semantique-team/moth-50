from pathlib import Path

import pytest

from training.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_base6_config_is_the_moth50_primary():
    config = load_config(ROOT / "training" / "configs" / "base6.toml")
    assert config.model.encoder() == {
        "width": 512,
        "n_layers": 6,
        "n_heads": 8,
        "ffn_dim": 2048,
    }
    assert config.model.head_blocks == 6
    assert config.model.checkpointing is True
    assert config.data.max_seconds == 192
    assert config.data.max_clips == 32
    assert len(config.data.train) == 6
    assert len(config.data.dev) == 2
    assert config.data.train[-1].repeat == 50
    assert config.data.tokenizer.is_file()
    assert all(source.manifest.is_absolute() for source in config.data.train)


def test_smoke_config_is_short_and_deterministic():
    config = load_config(ROOT / "training" / "configs" / "smoke.toml")
    assert config.data.augment is False
    assert config.epochs == 4
    assert config.max_steps == 20
    assert config.data.max_seconds == 10


def test_unknown_preset_is_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(
        """
[model]
preset = "small"
[data]
tokenizer = "x"
[[data.train]]
manifest = "m.jsonl"
shards = "s"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="preset"):
        load_config(path)


def test_missing_tokenizer_is_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(
        """
[model]
preset = "base"
[data]
[[data.train]]
manifest = "m.jsonl"
shards = "s"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tokenizer"):
        load_config(path)


def test_repeat_must_be_positive(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(
        """
[model]
preset = "base"
[data]
tokenizer = "x"
[[data.train]]
manifest = "m.jsonl"
shards = "s"
repeat = 0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="repeat"):
        load_config(path)
