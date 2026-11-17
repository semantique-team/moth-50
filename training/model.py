"""Model construction and the MOTH-50 training forward.

The model itself is Moth's shape-generic ``MothSTT``. This module only supplies
the MOTH-50 encoder config and an optional per-block checkpointed forward, so a
base+6 run is not silently built as Whisper-tiny.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from stt.model import EncoderConfig, MothSTT, _padding_mask
from torch.utils.checkpoint import checkpoint

from training.config import ModelConfig


def build_model(config: ModelConfig) -> MothSTT:
    encoder = EncoderConfig(**config.encoder())
    return MothSTT(
        vocab_size=config.vocab_size,
        encoder_config=encoder,
        head_blocks=config.head_blocks,
    )


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
