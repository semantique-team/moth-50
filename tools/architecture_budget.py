#!/usr/bin/env python3
"""Estimate parameter count, INT8 size, package size and compute for MOTH-50.

This is a planning tool, not an export measurement. It uses the same arithmetic
as ``docs/architecture-options.md``:

- encoder: conv stem (80 -> width, stride 2) + ``layers`` pre-LN blocks + final LN
- optional extra pre-head blocks at the same width
- linear CTC head over ``vocab`` BPE pieces plus blank
- fixed sinusoidal positions are buffers, not parameters

The INT8 estimate uses 1.03 bytes/parameter, the ratio observed in the Moth
0.8.1 export. The package adds the LM, the VAD and ~0.08 MB of misc files.
Throughput is extrapolated from Moth E-028 (342 audio-s/s at 1.06 GFLOP/s) with
a 10% pessimism factor. All numbers must be confirmed by a real export and a
200-step training probe.

Usage:
    python tools/architecture_budget.py
    python tools/architecture_budget.py --width 512 --layers 6 --ffn 2048 \
        --extra 6 --lm-mb 5.4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    name: str
    width: int
    layers: int
    ffn: int
    extra: int = 0
    vocab: int = 1024
    frame_rate: int = 50
    lm_mb: float = 5.4
    vad_mb: float = 2.33
    misc_mb: float = 0.08


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


def gflops_per_second(c: Candidate) -> float:
    """Forward MACs for one second of audio, counted as 2 FLOPs per MAC."""
    blocks = c.layers + c.extra
    per_frame = blocks * (4 * c.width * c.width + 2 * c.width * c.ffn)
    return per_frame * c.frame_rate * 2 / 1e9


def estimate(c: Candidate) -> dict[str, float]:
    params = acoustic_params(c)
    int8_mb = params * 1.03 / 1e6
    package_mb = int8_mb + c.lm_mb + c.vad_mb + c.misc_mb
    flops = gflops_per_second(c)
    # Moth E-028: 342 audio-s/s at 1.06 GFLOP/s; 10% pessimism factor.
    audio_s_per_s = 342.0 * (1.06 / flops) * 0.9
    hours_per_13k_steps = 13_283 * 192.0 / audio_s_per_s / 3600.0
    return {
        "params_m": params / 1e6,
        "int8_mb": int8_mb,
        "package_mb": package_mb,
        "gflops": flops,
        "flops_vs_moth": flops / 1.06,
        "audio_s_per_s": audio_s_per_s,
        "hours_per_epoch": hours_per_13k_steps,
    }


CANDIDATES = [
    Candidate("moth 0.8.1 (tiny+2)", 384, 4, 1536, extra=2),
    Candidate("base", 512, 6, 2048),
    Candidate("base +2", 512, 6, 2048, extra=2),
    Candidate("base +4", 512, 6, 2048, extra=4),
    Candidate("base +6", 512, 6, 2048, extra=6),
    Candidate("base +8", 512, 6, 2048, extra=8),
    Candidate("base +12", 512, 6, 2048, extra=12),
    Candidate("base +4, 12 MB LM", 512, 6, 2048, extra=4, lm_mb=12.0),
    Candidate("base +2, 20 MB LM", 512, 6, 2048, extra=2, lm_mb=20.0),
    Candidate("small", 768, 12, 3072),
    Candidate("small +2", 768, 12, 3072, extra=2),
]


def format_table(candidates: list[Candidate]) -> str:
    header = (
        f"{'candidate':24s} {'params':>8s} {'int8':>7s} {'package':>8s} "
        f"{'GFLOP/s':>8s} {'xMoth':>6s} {'audio-s/s':>9s} {'h/epoch':>8s}"
    )
    lines = [header, "-" * len(header)]
    for c in candidates:
        m = estimate(c)
        lines.append(
            f"{c.name:24s} {m['params_m']:7.2f}M {m['int8_mb']:6.1f}M "
            f"{m['package_mb']:7.1f}M {m['gflops']:8.2f} "
            f"{m['flops_vs_moth']:5.1f}x {m['audio_s_per_s']:9.1f} "
            f"{m['hours_per_epoch']:8.1f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--ffn", type=int, default=2048)
    parser.add_argument("--extra", type=int, default=6)
    parser.add_argument("--vocab", type=int, default=1024)
    parser.add_argument("--frame-rate", type=int, default=50)
    parser.add_argument("--lm-mb", type=float, default=5.4)
    parser.add_argument("--vad-mb", type=float, default=2.33)
    parser.add_argument("--all", action="store_true", help="print the candidate table")
    args = parser.parse_args()

    if args.all:
        print(format_table(CANDIDATES))
        return

    c = Candidate(
        "custom",
        args.width,
        args.layers,
        args.ffn,
        extra=args.extra,
        vocab=args.vocab,
        frame_rate=args.frame_rate,
        lm_mb=args.lm_mb,
        vad_mb=args.vad_mb,
    )
    m = estimate(c)
    print(f"candidate: {c}")
    for key, value in m.items():
        print(f"  {key:16s} {value:10.2f}")


if __name__ == "__main__":
    main()
