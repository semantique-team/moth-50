#!/usr/bin/env python3
"""Count the activation memory that MOTH-50 candidates keep for the backward pass.

Runs one bf16-autocast forward of Moth's ``stt.model.MothSTT`` (random weights,
equal-length clips, no padding) and sums the unique storages that autograd saves
for backward, per segment: conv stem, each transformer block, final norm + head.
This is the activation term of training memory; it excludes weights, gradients,
optimizer state, allocator caches and backward transients.

With per-block activation checkpointing, only the block inputs are kept and one
block is recomputed at a time, so the estimate is: stem + head + block inputs +
the largest single block.

``--device mps`` (or ``cuda``) additionally runs one real bf16 training step
(forward, backward, AdamW) and reports the allocator after each phase, with or
without ``--checkpointing``. On MPS the reading is
``torch.mps.driver_allocated_memory``, the quantity Moth logged (E-028 peaked at
7,495.7 MiB); on CUDA the true peak is reported too. Use ``--no-count`` on a
16 GB machine so the CPU count does not compete for the same unified memory.
This is the E-002 memory probe.

Usage:
    python tools/probe_activations.py --moth-stt ../moth-stt-dev \
        --out runs/probe-2026-11-17/activations.json
    python tools/probe_activations.py --moth-stt ../moth-stt --device mps --no-count \
        --candidates base+6 --batches 6x30 --checkpointing --out /tmp/mps.json

``--moth-stt`` is a Moth repository checkout with ``stt/`` (on the training
machine the frozen repository; never the published release folder).
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
from pathlib import Path

CANDIDATES = {
    "tiny+2": ("tiny", 2),
    "base+0": ("base", 0),
    "base+2": ("base", 2),
    "base+4": ("base", 4),
    "base+6": ("base", 6),
}
#: Batches of 192 audio-seconds (the E-028 cap) at three clip lengths.
BATCHES = {"6x30": (6, 30.0), "12x16": (12, 16.0), "32x6": (32, 6.0)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--moth-stt",
        type=Path,
        required=True,
        help="Moth repository checkout with stt/ (not the published release folder)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES))
    parser.add_argument("--batches", nargs="+", default=list(BATCHES))
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument(
        "--checkpointing",
        action="store_true",
        help="with --device mps/cuda: run the real step with per-block checkpointing",
    )
    parser.add_argument(
        "--no-count",
        action="store_true",
        help="skip the CPU saved-tensor count (use on a 16 GB machine with --device mps)",
    )
    args = parser.parse_args(argv)
    if args.no_count and args.device == "cpu":
        parser.error("--no-count needs --device mps or cuda")

    sys.path.insert(0, str(args.moth_stt.resolve()))
    import torch
    import torch.nn.functional as F
    from stt.model import EncoderConfig, MothSTT, _padding_mask
    from torch.utils.checkpoint import checkpoint

    encoders = {
        "tiny": EncoderConfig(),
        "base": EncoderConfig(width=512, n_layers=6, n_heads=8, ffn_dim=2048),
    }
    device = torch.device(args.device)

    class Recorder:
        def __init__(self) -> None:
            self.segment = "stem"
            self.saved: dict[str, dict[tuple[int, int], int]] = {}

        def pack(self, tensor):
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            self.saved.setdefault(self.segment, {})[key] = storage.nbytes()
            return tensor

    def forward(model, mel, lengths, recorder=None, use_checkpointing=False):
        """``MothSTT.forward`` unrolled so segments can be labelled or checkpointed."""

        def mark(name: str) -> None:
            if recorder is not None:
                recorder.segment = name

        def run(block, x, keep):
            if use_checkpointing:
                return checkpoint(block, x, keep, use_reentrant=False)
            return block(x, keep)

        encoder = model.encoder
        mark("stem")
        x = F.gelu(encoder.conv1(mel))
        x = F.gelu(encoder.conv2(x)).transpose(1, 2)
        enc_lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        x = x + encoder.positions[: x.shape[1]].to(x.dtype)
        keep = _padding_mask(enc_lengths, x.shape[1])
        boundary = x.element_size() * x.numel()
        for index, block in enumerate([*encoder.blocks, *model.head_blocks]):
            if index == len(encoder.blocks):
                mark("final_ln")
                x = encoder.final_ln(x)
            mark(f"block{index}")
            x = run(block, x, keep)
        if not model.head_blocks:
            mark("final_ln")
            x = encoder.final_ln(x)
        mark("head")
        return F.log_softmax(model.head(x).float(), dim=-1), boundary

    def count_saved(model, mel, lengths, n_blocks: int) -> dict:
        """CPU bf16 forward: unique storages saved for backward, per segment."""
        recorder = Recorder()
        with (
            torch.autograd.graph.saved_tensors_hooks(recorder.pack, lambda t: t),
            torch.autocast("cpu", dtype=torch.bfloat16),
        ):
            out, boundary = forward(model, mel, lengths, recorder)
        del out
        unique: dict[tuple[int, int], int] = {}
        for entries in recorder.saved.values():
            unique.update(entries)
        per_segment = {seg: sum(v.values()) for seg, v in recorder.saved.items()}
        blocks = [per_segment.get(f"block{i}", 0) for i in range(n_blocks)]
        outside = sum(v for seg, v in per_segment.items() if not seg.startswith("block"))
        return {
            "saved_gb": round(sum(unique.values()) / 1e9, 3),
            "largest_block_gb": round(max(blocks) / 1e9, 3),
            "checkpointing_estimate_gb": round(
                (outside + n_blocks * boundary + max(blocks)) / 1e9, 3
            ),
        }

    def memory() -> int:
        """Allocator reading comparable to Moth's training log (MPS: driver memory)."""
        if device.type == "mps":
            torch.mps.synchronize()
            return int(torch.mps.driver_allocated_memory())
        torch.cuda.synchronize()
        return int(torch.cuda.memory_allocated())

    def device_step(model, mel, lengths) -> dict:
        """One real bf16 training step (forward, backward, AdamW) on the device."""
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        if device.type == "mps":
            torch.mps.empty_cache()
        else:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        readings = {}
        with torch.autocast(device.type, dtype=torch.bfloat16):
            out, _ = forward(
                model, mel.to(device), lengths.to(device), use_checkpointing=args.checkpointing
            )
        readings["after_forward"] = memory()
        out.float().mean().backward()
        readings["after_backward"] = memory()
        optimizer.step()
        readings["after_step"] = memory()
        result = {
            f"{device.type}_checkpointing": args.checkpointing,
            **{f"{device.type}_{key}_gb": round(value / 1e9, 3) for key, value in readings.items()},
            f"{device.type}_max_observed_gb": round(max(readings.values()) / 1e9, 3),
        }
        if device.type == "cuda":
            result["cuda_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        del out, optimizer
        model.zero_grad(set_to_none=True)
        model.to("cpu")
        return result

    rows = []
    for name in args.candidates:
        encoder_name, extra = CANDIDATES[name]
        torch.manual_seed(0)
        model = MothSTT(
            vocab_size=1024, encoder_config=encoders[encoder_name], head_blocks=extra
        ).train()
        n_blocks = len(model.encoder.blocks) + len(model.head_blocks)
        for batch_name in args.batches:
            clips, seconds = BATCHES[batch_name]
            frames = int(seconds * 100)
            mel = torch.randn(clips, 80, frames) * 0.1
            lengths = torch.full((clips,), frames)

            row: dict = {"candidate": name, "batch": batch_name}
            if not args.no_count:
                row.update(count_saved(model, mel, lengths, n_blocks))
            if device.type != "cpu":
                row.update(device_step(model, mel, lengths))

            rows.append(row)
            print(json.dumps(row), flush=True)
            gc.collect()

    report = {
        "tool": "tools/probe_activations.py",
        "definition": "unique storages saved for backward, bf16 autocast, equal-length clips",
        "platform": platform.platform(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
