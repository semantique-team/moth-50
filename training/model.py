"""Model construction and the MOTH-50 training forward.

The model itself is Moth's shape-generic ``MothSTT``. This module only supplies
the MOTH-50 encoder config and an optional per-block checkpointed forward, so a
base+6 run is not silently built as Whisper-tiny.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from stt.model import EncoderConfig, MothSTT, _padding_mask
from torch.utils.checkpoint import checkpoint

from training.config import ModelConfig


def build_model(config: ModelConfig) -> MothSTT:
    encoder = EncoderConfig(**config.encoder())
    model = MothSTT(
        vocab_size=config.vocab_size,
        encoder_config=encoder,
        head_blocks=config.head_blocks,
    )
    if config.init is not None:
        load_encoder_init(model, config.init)
    if config.identity_heads:
        identity_heads(model)
    return model


def load_encoder_init(model: MothSTT, path: Path) -> None:
    """Load a converted Whisper encoder produced by ``tools/convert_whisper_encoder.py``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"model.init not found: {path}; build it with "
            "tools/convert_whisper_encoder.py (see training/README.md)"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = payload.get("config") or {}
    expected = {
        "width": model.encoder.config.width,
        "n_layers": model.encoder.config.n_layers,
        "n_heads": model.encoder.config.n_heads,
        "ffn_dim": model.encoder.config.ffn_dim,
    }
    mismatch = {
        name: (saved[name], value)
        for name, value in expected.items()
        if name in saved and saved[name] != value
    }
    if mismatch:
        raise ValueError(f"model.init {path} was built for a different encoder: {mismatch}")
    model.encoder.load_state_dict(payload["state_dict"], strict=True)


def identity_heads(model: MothSTT) -> None:
    """Zero ``out``/``mlp2`` of every extra block: each block is an exact identity at step 0."""
    for block in model.head_blocks:
        for layer in (block.out, block.mlp2):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)


def forward_checkpointed(
    model: MothSTT, mel: torch.Tensor, mel_lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``MothSTT.forward`` unrolled so each transformer block is recomputed."""
    encoder = model.encoder
    x = F.gelu(encoder.conv1(mel))
    x = F.gelu(encoder.conv2(x)).transpose(1, 2)
    lengths = torch.div(mel_lengths + 1, 2, rounding_mode="floor")
    x = x + encoder.positions[: x.shape[1]].to(x.dtype)
    keep = _padding_mask(lengths, x.shape[1])
    for block in encoder.blocks:
        x = checkpoint(block, x, keep, use_reentrant=False)
    x = encoder.final_ln(x)
    if model.head_blocks:
        keep = _padding_mask(lengths, x.shape[1])
        for block in model.head_blocks:
            x = checkpoint(block, x, keep, use_reentrant=False)
    logits = model.head(x)
    return F.log_softmax(logits.float(), dim=-1), lengths


def forward(
    model: MothSTT,
    mel: torch.Tensor,
    mel_lengths: torch.Tensor,
    *,
    checkpointing: bool,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if checkpointing and training:
        return forward_checkpointed(model, mel, mel_lengths)
    return model(mel, mel_lengths)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def memory_gb(device: torch.device) -> float:
    """Allocator reading comparable to Moth's MPS training log."""
    synchronize(device)
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def empty_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
