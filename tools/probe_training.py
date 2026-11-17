#!/usr/bin/env python3
"""Run a real 200-step MPS training probe for a MOTH-50 candidate.

This is the E-002 memory/stability probe, not a quality experiment. It builds a
randomly initialized ``stt.model.MothSTT``, loads real ``moth-cs`` training
clips through Moth's own 16 kHz / log-mel path, tokenizes the transcripts with
the pinned ``moth_bpe.model``, and runs real bf16 forward/backward/AdamW steps
with CTC loss on MPS. The 6 x 30 s bucket is padded to the 30 s cap, so the
attention memory is the worst case rather than the mean clip length.

It reports peak MPS driver memory, step time and padded-audio throughput with or
without per-block activation checkpointing, and checks for non-finite losses. It
does not save checkpoints and does not include augmentation or the final trainer
port.

Usage:
    uv run --frozen --project ../moth-stt --extra training -- \
      python tools/probe_training.py --moth-stt ../moth-stt \
      --manifest ../moth-cs-dataset/manifests/train.jsonl \
      --audio-root ../moth-cs-dataset \
      --tokenizer data/tokenizer/moth_bpe.model \
      --steps 200 --batch-clips 6 --device mps --out runs/E-002/train-probe.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

CANDIDATES = {
    "base+6": {
        "encoder": {"width": 512, "n_layers": 6, "n_heads": 8, "ffn_dim": 2048},
        "head_blocks": 6,
    }
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--moth-stt", type=Path, required=True, help="Moth checkout with stt/ (not release/)"
    )
    parser.add_argument("--manifest", type=Path, required=True, help="moth-cs train manifest")
    parser.add_argument("--audio-root", type=Path, required=True, help="root of manifest paths")
    parser.add_argument("--tokenizer", type=Path, required=True, help="pinned moth_bpe.model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), default="base+6")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10, help="steps excluded from timing")
    parser.add_argument("--batch-clips", type=int, default=6)
    parser.add_argument(
        "--max-seconds", type=float, default=30.0, help="pad every batch to this cap"
    )
    parser.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--empty-cache-every", type=int, default=10)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="use only the first N clips (0 = all)")
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.batch_clips <= 0:
        parser.error("--steps and --batch-clips must be positive")

    sys.path.insert(0, str(args.moth_stt.resolve()))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from stt.data.build_mel import load_audio_16k
    from stt.features import log_mel
    from stt.model import EncoderConfig, MothSTT, _padding_mask
    from stt.tokenizer import encode as tokenizer_encode
    from stt.tokenizer import load_tokenizer
    from torch.utils.checkpoint import checkpoint

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("norm", "").strip()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("manifest has no usable rows")

    tokenizer = load_tokenizer(args.tokenizer)
    mels: list[np.ndarray] = []
    durations: list[float] = []
    targets: list[list[int] | None] = []
    skipped = 0
    print(f"loading {len(rows)} clips from {args.manifest}", flush=True)
    for index, row in enumerate(rows):
        wav = load_audio_16k(args.audio_root / row["audio"])
        mel = log_mel(wav).astype(np.float32)
        ids = tokenizer_encode(tokenizer, row["norm"])
        if mel.shape[1] > int(args.max_seconds * 100) or len(ids) > (mel.shape[1] + 1) // 2:
            skipped += 1
            targets.append(None)
        else:
            targets.append(ids)
        mels.append(mel)
        durations.append(len(wav) / 16000.0)
        if (index + 1) % 200 == 0:
            print(f"  {index + 1}/{len(rows)}", flush=True)
    if not any(target is not None for target in targets):
        raise SystemExit("no clip fits the CTC length constraint")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    spec = CANDIDATES[args.candidate]
    encoder = EncoderConfig(**spec["encoder"])
    model = MothSTT(
        vocab_size=1024, encoder_config=encoder, head_blocks=spec["head_blocks"]
    ).train()
    params = sum(parameter.numel() for parameter in model.parameters())
    device = torch.device(args.device)
    model.to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr_encoder},
            {
                "params": [*model.head_blocks.parameters(), *model.head.parameters()],
                "lr": args.lr_head,
            },
        ],
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01,
    )
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    padded_frames = int(args.max_seconds * 100)

    def memory_gb() -> float:
        if device.type == "mps":
            torch.mps.synchronize()
            return torch.mps.driver_allocated_memory() / 1e9
        if device.type == "cuda":
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / 1e9
        return 0.0

    def sync() -> None:
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    def forward_checkpointed(model, mel: torch.Tensor, mel_lengths: torch.Tensor):
        """MothSTT.forward unrolled so each transformer block can be recomputed."""
        encoder_module = model.encoder
        x = F.gelu(encoder_module.conv1(mel))
        x = F.gelu(encoder_module.conv2(x)).transpose(1, 2)
        enc_lengths = torch.div(mel_lengths + 1, 2, rounding_mode="floor")
        x = x + encoder_module.positions[: x.shape[1]].to(x.dtype)
        keep = _padding_mask(enc_lengths, x.shape[1])
        for block in encoder_module.blocks:
            x = checkpoint(block, x, keep, use_reentrant=False)
        x = encoder_module.final_ln(x)
        if model.head_blocks:
            keep = _padding_mask(enc_lengths, x.shape[1])
            for block in model.head_blocks:
                x = checkpoint(block, x, keep, use_reentrant=False)
        logits = model.head(x)
        return F.log_softmax(logits.float(), dim=-1), enc_lengths

    def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        usable = [index for index, target in enumerate(targets) if target is not None]
        chosen = rng.choice(usable, size=args.batch_clips, replace=True)
        mel = torch.zeros(args.batch_clips, 80, padded_frames, dtype=torch.float32)
        lengths = torch.zeros(args.batch_clips, dtype=torch.long)
        flat: list[int] = []
        target_lengths = torch.zeros(args.batch_clips, dtype=torch.long)
        for position, index in enumerate(chosen):
            clip = mels[int(index)]
            frames = min(clip.shape[1], padded_frames)
            mel[position, :, :frames] = torch.from_numpy(np.ascontiguousarray(clip[:, :frames]))
            lengths[position] = frames
            ids = targets[int(index)] or []
            flat.extend(ids)
            target_lengths[position] = len(ids)
        target_tensor = torch.tensor(flat, dtype=torch.long)
        return (
            mel.to(device),
            lengths.to(device),
            target_tensor.to(device),
            target_lengths.to(device),
        )

    losses: list[float] = []
    timings: list[float] = []
    memories: list[float] = []
    nonfinite = 0
    first_breakdown: dict[str, float] = {}
    print(
        f"running {args.steps} steps: {args.candidate}, batch {args.batch_clips}x"
        f"{args.max_seconds:g}s, checkpointing={args.checkpointing}, device={args.device}",
        flush=True,
    )
    for step in range(args.steps):
        mel, lengths, target, target_lengths = batch()
        optimizer.zero_grad(set_to_none=True)
        sync()
        started = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16):
            if args.checkpointing:
                log_probs, out_lengths = forward_checkpointed(model, mel, lengths)
            else:
                log_probs, out_lengths = model(mel, lengths)
            loss = F.ctc_loss(
                log_probs.transpose(0, 1),
                target,
                out_lengths,
                target_lengths,
                blank=0,
                zero_infinity=True,
            )
        if step == 0:
            first_breakdown["forward_gb"] = round(memory_gb(), 3)
        finite = bool(torch.isfinite(loss))
        if finite:
            loss.backward()
            if step == 0:
                first_breakdown["backward_gb"] = round(memory_gb(), 3)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        else:
            nonfinite += 1
        sync()
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        reading = memory_gb()
        memories.append(reading)
        if step == 0:
            first_breakdown["step_gb"] = round(reading, 3)
        if (
            args.empty_cache_every
            and (step + 1) % args.empty_cache_every == 0
            and device.type == "mps"
        ):
            torch.mps.empty_cache()
        if (step + 1) % 50 == 0 or step + 1 == args.steps:
            print(
                f"  step {step + 1}/{args.steps}  loss={losses[-1] if losses else float('nan'):.4f}"
                f"  mem={reading:.2f} GB  step={elapsed:.3f} s",
                flush=True,
            )

    timed = timings[args.warmup :] or timings
    report = {
        "tool": "tools/probe_training.py",
        "definition": (
            "real moth-cs mels + pinned tokenizer, random-init MothSTT, bf16 autocast, "
            "fp32 CTC + AdamW, batch padded to the 30 s cap; no augmentation, no checkpoint"
        ),
        "platform": platform.platform(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "device": args.device,
        "candidate": args.candidate,
        "encoder": spec["encoder"],
        "head_blocks": spec["head_blocks"],
        "params": params,
        "checkpointing": bool(args.checkpointing),
        "steps": args.steps,
        "warmup": args.warmup,
        "batch_clips": args.batch_clips,
        "max_seconds": args.max_seconds,
        "padded_frames": padded_frames,
        "data": {
            "manifest": str(args.manifest),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "clips_loaded": len(rows),
            "clips_skipped": skipped,
            "mean_clip_seconds": round(statistics.fmean(durations), 3),
            "max_clip_seconds": round(max(durations), 3),
            "padded_audio_seconds_per_step": args.batch_clips * args.max_seconds,
        },
        "tokenizer": {
            "path": str(args.tokenizer),
            "sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        },
        "optimizer": {
            "name": "AdamW",
            "lr_head": args.lr_head,
            "lr_encoder": args.lr_encoder,
            "betas": [0.9, 0.98],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
        },
        "memory_gb": {
            "first_step": first_breakdown,
            "max_observed": round(max(memories), 3),
            "last": round(memories[-1], 3),
        },
        "timing_s": {
            "median": round(statistics.median(timed), 4),
            "mean": round(statistics.fmean(timed), 4),
            "total": round(sum(timings), 2),
        },
        "throughput": {
            "padded_audio_s_per_s": round(
                args.batch_clips * args.max_seconds / statistics.fmean(timed), 2
            ),
            "steps_per_s": round(1 / statistics.fmean(timed), 4),
        },
        "loss": {
            "first": round(losses[0], 4) if losses else None,
            "last": round(losses[-1], 4) if losses else None,
            "min": round(min(losses), 4) if losses else None,
            "max": round(max(losses), 4) if losses else None,
            "nonfinite_steps": nonfinite,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    del model, optimizer
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
