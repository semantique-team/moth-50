#!/usr/bin/env python3
"""Estimate parameter count, INT8 size, package size and compute for MOTH-50.

This is a planning tool, not an export measurement. It uses the same arithmetic
as ``docs/architecture-options.md``:

- encoder: conv stem (80 -> width, stride 2) + ``layers`` pre-LN blocks + final LN
- optional extra pre-head blocks at the same width
- linear CTC head over ``vocab`` BPE pieces plus blank
- fixed sinusoidal positions are generated in the graph, not parameters

INT8 bytes per parameter are measured, not assumed: Moth's ``quantize_dynamic``
recipe (per-channel QInt8, ORT 1.31) on random-weight exports gives 1.0260 B/param
at width 384 and 1.0175 B/param at width 512 (``runs/probe-2026-11-17``). Other
widths fall back to the conservative 1.03. The package adds the LM, the VAD and
the misc files of the Moth 0.8.1 release (vocab, mel filters, config, NOTICE).
The budget is decimal bytes, as in Moth (D5: 50,000,000 B).

Throughput is extrapolated from Moth E-028 by FLOPs with a 10% pessimism factor:
332.7 audio-s per wall-second (time-weighted mean of ``train.log.jsonl``, 590.8 h
in 6,393 s) for the tiny+2 model, whose blocks cost 1.06 GFLOP per audio-second.
An E-028 epoch is 590.8 h of audio (13,283 steps, ~160 audio-s per step; 192 s is
the batch cap, not the mean). Activation checkpointing adds one forward pass,
~4/3 of the step time. All numbers must be confirmed by a real export and a
training probe on the target machine.

Usage:
    python tools/architecture_budget.py --all
    python tools/architecture_budget.py --width 512 --layers 6 --ffn 2048 \
        --extra 6 --lm-mb 5.381992
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

BUDGET_BYTES = 50_000_000
LM_BYTES = 3_353_840  # E-012(b) MOTHLMB2 A3 (lossless 10-bit codebook)
VAD_BYTES = 1_289_603  # Silero VAD v5 16 kHz-only graph (D19)
MISC_BYTES = 86_993  # vocab.json + mel_filters.bin + config.json + NOTICE (E-012 package)

#: Measured INT8 bytes per parameter by model width (random-weight exports).
MEASURED_BYTES_PER_PARAM = {384: 1.0260, 512: 1.0175}
FALLBACK_BYTES_PER_PARAM = 1.03

#: Moth E-028 anchor: tiny+2 throughput and epoch length.
ANCHOR_AUDIO_S_PER_S = 332.7
EPOCH_AUDIO_HOURS = 590.8
PESSIMISM = 0.9
CHECKPOINTING_FACTOR = 4 / 3


@dataclass(frozen=True)
class Candidate:
    name: str
    width: int
    layers: int
    ffn: int
    extra: int = 0
    vocab: int = 1024
    frame_rate: int = 50
    lm_bytes: int = LM_BYTES
    vad_bytes: int = VAD_BYTES
    misc_bytes: int = MISC_BYTES


def block_params(width: int, ffn: int) -> int:
    """Pre-LN transformer block: attention + 2 LayerNorms + MLP."""
    attention = 4 * width * width + 3 * width  # q, v, out have biases; k does not
    layer_norms = 4 * width
    mlp = width * ffn + ffn + ffn * width + width
    return attention + layer_norms + mlp


def acoustic_params(c: Candidate) -> int:
    conv = 80 * c.width * 3 + c.width + c.width * c.width * 3 + c.width
    final_norm = 2 * c.width
    head = c.width * (c.vocab + 1) + (c.vocab + 1)
    return (
        conv
        + final_norm
        + c.layers * block_params(c.width, c.ffn)
        + c.extra * block_params(c.width, c.ffn)
        + head
    )


def bytes_per_param(width: int) -> tuple[float, bool]:
    """INT8 bytes per parameter and whether the ratio was measured at this width."""
    if width in MEASURED_BYTES_PER_PARAM:
        return MEASURED_BYTES_PER_PARAM[width], True
    return FALLBACK_BYTES_PER_PARAM, False


def gflops_per_second(c: Candidate) -> float:
    """Block MACs for one second of audio, counted as 2 FLOPs per MAC.

    Excludes the conv stem, the head and the quadratic attention term; the
    throughput anchor was measured with the same definition.
    """
    blocks = c.layers + c.extra
    per_frame = blocks * (4 * c.width * c.width + 2 * c.width * c.ffn)
    return per_frame * c.frame_rate * 2 / 1e9


ANCHOR = Candidate("moth 0.8.1 (tiny+2)", 384, 4, 1536, extra=2)


def estimate(c: Candidate, budget_bytes: int = BUDGET_BYTES) -> dict[str, float]:
    params = acoustic_params(c)
    ratio, measured = bytes_per_param(c.width)
    int8_bytes = params * ratio
    package_bytes = int8_bytes + c.lm_bytes + c.vad_bytes + c.misc_bytes
    flops = gflops_per_second(c)
    audio_s_per_s = ANCHOR_AUDIO_S_PER_S * (gflops_per_second(ANCHOR) / flops) * PESSIMISM
    hours_per_epoch = EPOCH_AUDIO_HOURS / audio_s_per_s
    return {
        "params_m": params / 1e6,
        "int8_mb": int8_bytes / 1e6,
        "int8_ratio_measured": float(measured),
        "package_mb": package_bytes / 1e6,
        "headroom_mb": (budget_bytes - package_bytes) / 1e6,
        "max_lm_mb": (budget_bytes - int8_bytes - c.vad_bytes - c.misc_bytes) / 1e6,
        "fits": float(package_bytes <= budget_bytes),
        "gflops": flops,
        "flops_vs_moth": flops / gflops_per_second(ANCHOR),
        "audio_s_per_s": audio_s_per_s,
        "hours_per_epoch": hours_per_epoch,
        "hours_per_epoch_checkpointing": hours_per_epoch * CHECKPOINTING_FACTOR,
    }


CANDIDATES = [
    ANCHOR,
    Candidate("base", 512, 6, 2048),
    Candidate("base +2", 512, 6, 2048, extra=2),
    Candidate("base +4", 512, 6, 2048, extra=4),
    Candidate("base +6", 512, 6, 2048, extra=6),
    Candidate("base +7", 512, 6, 2048, extra=7),
    Candidate("base +8", 512, 6, 2048, extra=8),
    Candidate("base +4, 12 MB LM", 512, 6, 2048, extra=4, lm_bytes=12_000_000),
    Candidate("base +2, 20 MB LM", 512, 6, 2048, extra=2, lm_bytes=20_000_000),
    Candidate("small (rejected)", 768, 12, 3072),
]


def format_table(candidates: list[Candidate], budget_bytes: int = BUDGET_BYTES) -> str:
    header = (
        f"{'candidate':22s} {'params':>8s} {'int8':>7s} {'package':>8s} {'room':>6s} "
        f"{'maxLM':>6s} {'GFLOP/s':>8s} {'xMoth':>6s} {'audio-s/s':>9s} {'h/ep':>5s} "
        f"{'h/ep ckpt':>9s}"
    )
    lines = [header, "-" * len(header)]
    for c in candidates:
        m = estimate(c, budget_bytes)
        mark = "" if m["int8_ratio_measured"] else "*"
        lines.append(
            f"{c.name:22s} {m['params_m']:7.2f}M {m['int8_mb']:6.2f}{mark:1s}"
            f"{m['package_mb']:8.2f} {m['headroom_mb']:6.2f} {m['max_lm_mb']:6.2f} "
            f"{m['gflops']:8.2f} {m['flops_vs_moth']:5.1f}x {m['audio_s_per_s']:9.1f} "
            f"{m['hours_per_epoch']:5.1f} {m['hours_per_epoch_checkpointing']:9.1f}"
        )
    lines.append(
        f"MB = 1e6 bytes; budget {budget_bytes:,} B; room = budget - package; "
        "* = INT8 ratio not measured at this width"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--all", action="store_true", help="print the candidate table")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--ffn", type=int, default=2048)
    parser.add_argument("--extra", type=int, default=6)
    parser.add_argument("--vocab", type=int, default=1024)
    parser.add_argument("--frame-rate", type=int, default=50)
    parser.add_argument("--lm-mb", type=float, default=LM_BYTES / 1e6)
    parser.add_argument("--vad-mb", type=float, default=VAD_BYTES / 1e6)
    parser.add_argument("--misc-mb", type=float, default=MISC_BYTES / 1e6)
    parser.add_argument("--budget-bytes", type=int, default=BUDGET_BYTES)
    args = parser.parse_args(argv)

    if args.all:
        print(format_table(CANDIDATES, args.budget_bytes))
        return

    c = Candidate(
        "custom",
        args.width,
        args.layers,
        args.ffn,
        extra=args.extra,
        vocab=args.vocab,
        frame_rate=args.frame_rate,
        lm_bytes=round(args.lm_mb * 1e6),
        vad_bytes=round(args.vad_mb * 1e6),
        misc_bytes=round(args.misc_mb * 1e6),
    )
    print(f"candidate: {c}")
    for key, value in estimate(c, args.budget_bytes).items():
        print(f"  {key:30s} {value:10.2f}")


if __name__ == "__main__":
    main()
