import dataclasses

import pytest

pytest.importorskip("torch")

import torch
from stt.model import EncoderConfig, WhisperTinyEncoder

from training.config import ModelConfig
from training.model import build_model


def tiny(**overrides) -> ModelConfig:
    return ModelConfig(
        preset="tiny",
        head_blocks=2,
        vocab_size=64,
        checkpointing=False,
        overrides={},
        **overrides,
    )


def test_extra_blocks_start_as_exact_identity():
    model = build_model(tiny())
    x = torch.randn(2, 7, 384)
    for block in model.head_blocks:
        assert torch.count_nonzero(block.out.weight) == 0
        assert torch.count_nonzero(block.out.bias) == 0
        assert torch.count_nonzero(block.mlp2.weight) == 0
        assert torch.count_nonzero(block.mlp2.bias) == 0
        assert torch.equal(block(x, None), x)


def test_identity_heads_can_be_disabled():
    model = build_model(tiny(identity_heads=False))
    assert any(torch.count_nonzero(block.out.weight) for block in model.head_blocks)


def test_encoder_init_loads_strictly(tmp_path):
    encoder = WhisperTinyEncoder(EncoderConfig())
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.add_(0.01)
    path = tmp_path / "init.pt"
    torch.save(
        {"state_dict": encoder.state_dict(), "config": dataclasses.asdict(EncoderConfig())},
        path,
    )
    model = build_model(tiny(init=path, identity_heads=False))
    assert torch.equal(model.encoder.blocks[2].key.weight, encoder.blocks[2].key.weight)
    assert torch.equal(model.encoder.final_ln.weight, encoder.final_ln.weight)


def test_encoder_init_mismatch_is_rejected(tmp_path):
    encoder = WhisperTinyEncoder(EncoderConfig(width=512, n_layers=6, n_heads=8, ffn_dim=2048))
    path = tmp_path / "init.pt"
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "config": dataclasses.asdict(
                EncoderConfig(width=512, n_layers=6, n_heads=8, ffn_dim=2048)
            ),
        },
        path,
    )
    with pytest.raises(ValueError, match="different encoder"):
        build_model(tiny(init=path))


def test_missing_init_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="convert_whisper_encoder"):
        build_model(tiny(init=tmp_path / "missing.pt"))


def _trainer_checkpoint(tmp_path):
    from training.model import file_sha256

    raw = build_model(tiny(identity_heads=False))
    ema = build_model(tiny(identity_heads=False))
    path = tmp_path / "ckpt_best.pt"
    torch.save(
        {"model": raw.state_dict(), "ema_model": ema.state_dict(), "global_step": 40000},
        path,
    )
    return path, file_sha256(path), raw, ema


def test_finetune_init_loads_the_pinned_ema_weights(tmp_path):
    from training.config import FinetuneConfig
    from training.model import load_finetune_init

    path, sha, _raw, ema = _trainer_checkpoint(tmp_path)
    model = build_model(tiny())
    meta = load_finetune_init(model, FinetuneConfig(init_checkpoint=path, sha256=sha))

    assert meta == {
        "init_checkpoint": str(path),
        "sha256": sha,
        "weights": "ema",
        "source_global_step": 40000,
    }
    for name, value in ema.state_dict().items():
        assert torch.equal(model.state_dict()[name], value)


def test_finetune_init_refuses_a_different_file(tmp_path):
    from training.config import FinetuneConfig
    from training.model import load_finetune_init

    path, _sha, _raw, _ema = _trainer_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="pinned"):
        load_finetune_init(build_model(tiny()), FinetuneConfig(path, sha256="0" * 64))
    with pytest.raises(FileNotFoundError, match="init_checkpoint"):
        load_finetune_init(build_model(tiny()), FinetuneConfig(tmp_path / "missing.pt"))
