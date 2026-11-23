from dataclasses import replace
from pathlib import Path

import pytest

from training.config import load_config, override_seed

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
    assert config.model.init == ROOT / "data" / "raw" / "whisper-base-encoder.pt"
    assert config.model.identity_heads is True
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


def test_seed_override_moves_every_seed():
    config = load_config(ROOT / "training" / "configs" / "base6.toml")
    moved = override_seed(config, 7)
    assert moved.seed == 7
    assert moved.data.seed == 7
    assert moved.model == config.model
    assert moved.data.train == config.data.train


def test_long_config_is_the_e005_schedule():
    config = load_config(ROOT / "training" / "configs" / "base6-long.toml")
    assert config.model.head_blocks == 6
    assert config.model.identity_heads is True
    assert config.model.init == ROOT / "data" / "raw" / "whisper-base-encoder.pt"
    assert config.schedule.warmup_steps == 1000
    assert config.eval_every_steps == 2000
    assert config.checkpoint_every_steps == 100
    assert len(config.data.train) == 6
    assert config.ema_decay == 0.0


def test_e010_config_only_moves_the_encoder_lr():
    base = load_config(ROOT / "training" / "configs" / "base6.toml")
    screen = load_config(ROOT / "training" / "configs" / "base6-enc1e-4.toml")
    assert screen.optimizer.lr_encoder == 1e-4
    assert base.optimizer.lr_encoder == 3e-5
    assert replace(screen, path=base.path, optimizer=base.optimizer) == base


def _with_train(tmp_path, train: str) -> Path:
    path = tmp_path / "ema.toml"
    path.write_text(
        f"""
[model]
preset = "base"
[data]
tokenizer = "x"
[[data.train]]
manifest = "m.jsonl"
shards = "s"
[train]
{train}
""",
        encoding="utf-8",
    )
    return path


def test_ema_decay_is_off_by_default_and_validated(tmp_path):
    assert load_config(_with_train(tmp_path, "seed = 0")).ema_decay == 0.0
    assert load_config(_with_train(tmp_path, "ema_decay = 0.9998")).ema_decay == 0.9998
    with pytest.raises(ValueError, match="ema_decay"):
        load_config(_with_train(tmp_path, "ema_decay = 1.0"))


def _with_finetune(tmp_path, model: str, finetune: str) -> Path:
    path = tmp_path / "finetune.toml"
    path.write_text(
        f"""
[model]
preset = "base"
{model}
[data]
tokenizer = "x"
[[data.train]]
manifest = "m.jsonl"
shards = "s"
[finetune]
{finetune}
""",
        encoding="utf-8",
    )
    return path


def test_finetune_is_off_by_default():
    assert load_config(ROOT / "training" / "configs" / "base6.toml").finetune is None


def test_finetune_table_is_parsed(tmp_path):
    sha = "ab" * 32
    config = load_config(
        _with_finetune(tmp_path, "", f'init_checkpoint = "ckpt.pt"\nsha256 = "{sha}"')
    )
    assert config.finetune.init_checkpoint == tmp_path / "ckpt.pt"
    assert config.finetune.sha256 == sha
    assert config.finetune.weights == "ema"


def test_finetune_rejects_model_init_bad_sha_and_weights(tmp_path):
    with pytest.raises(ValueError, match="replaces model.init"):
        load_config(_with_finetune(tmp_path, 'init = "w.pt"', 'init_checkpoint = "c.pt"'))
    with pytest.raises(ValueError, match="sha256"):
        load_config(_with_finetune(tmp_path, "", 'init_checkpoint = "c.pt"\nsha256 = "abc"'))
    with pytest.raises(ValueError, match="weights"):
        load_config(_with_finetune(tmp_path, "", 'init_checkpoint = "c.pt"\nweights = "best"'))
    with pytest.raises(ValueError, match="init_checkpoint"):
        load_config(_with_finetune(tmp_path, "", 'weights = "ema"'))


def test_e027_arm_and_control_differ_only_by_the_tts_source():
    control = load_config(ROOT / "training" / "configs" / "base6-e027-control.toml")
    arm = load_config(ROOT / "training" / "configs" / "base6-e027-tts.toml")
    r1 = load_config(ROOT / "training" / "configs" / "base6-e023-r1.toml")

    assert arm.data.train[:-1] == control.data.train == r1.data.train
    assert arm.data.train[-1].manifest.name == "tts_e027.jsonl"
    assert replace(arm, path=control.path, data=control.data) == control
    assert control.model.init is None
    assert control.finetune.init_checkpoint.name == "ckpt_best.pt"
    assert control.finetune.init_checkpoint.parent.parent.name == "r1-s0"
    assert control.finetune.weights == "ema"
    assert control.max_steps == 8000
    assert control.optimizer.lr_encoder == 1e-5
    assert control.optimizer.lr_head == 1e-4
