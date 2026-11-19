"""Convert a Hugging Face Whisper encoder into Moth's ``init`` format.

The MOTH-50 trainer starts every E-003 arm from the same pretrained Whisper
encoder (``docs/architecture-options.md`` §5.2); this tool produces that file.
It reuses Moth's own ``stt.convert_whisper`` mapping and parity check instead of
re-implementing the name map.

    cd ../moth-stt
    uv run --frozen --project . -- python ../moth-50/tools/convert_whisper_encoder.py \
      --model openai/whisper-base \
      --out ../moth-50/data/raw/whisper-base-encoder.pt

The output is a ``torch.save`` dict with ``state_dict`` (Moth encoder names),
``config`` (width/layers/heads/FFN), the HF revision and the measured parity.
``data/raw`` is gitignored: rerun this after a fresh clone.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import warnings
from pathlib import Path

import torch
from stt.convert_whisper import _PARITY_GATE, _POSITIONAL, convert_encoder
from stt.model import EncoderConfig


def config_for(hf_config) -> EncoderConfig:
    """Moth encoder config from an HF WhisperConfig (tiny, base, small, ...)."""
    return EncoderConfig(
        n_mels=int(hf_config.num_mel_bins),
        width=int(hf_config.d_model),
        n_layers=int(hf_config.encoder_layers),
        n_heads=int(hf_config.encoder_attention_heads),
        ffn_dim=int(hf_config.encoder_ffn_dim),
        max_frames=int(hf_config.max_source_positions),
    )


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].to(torch.float32).numpy().tobytes())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="convert_whisper_encoder.py", description=__doc__)
    parser.add_argument("--model", default="openai/whisper-base")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--skip-verify", action="store_true", help="write even if parity fails")
    args = parser.parse_args(argv)

    from transformers import WhisperModel

    warnings.filterwarnings("ignore", category=UserWarning)
    hf_model = WhisperModel.from_pretrained(args.model, attn_implementation="eager")
    hf_encoder = hf_model.encoder.eval()
    config = config_for(hf_model.config)
    revision = getattr(hf_model.config, "_commit_hash", None) or getattr(
        hf_model, "_commit_hash", None
    )

    encoder, report = convert_encoder(hf_encoder.state_dict(), config)
    encoder.eval()
    print(
        f"mapped: {report['mapped']} | skipped: {report['skipped']} | unknown: {report['unknown']}"
    )
    print(f"position table vs generated sinusoids: max |delta| = {report['position_max_diff']:.3e}")

    torch.manual_seed(0)
    mel = torch.randn(1, config.n_mels, config.max_frames * 2) * 0.1
    length = torch.tensor([config.max_frames * 2])
    with torch.no_grad():
        expected = hf_encoder(mel).last_hidden_state
        generated, _ = encoder(mel, length)
        table = hf_encoder.state_dict()[_POSITIONAL].to(torch.float32)
        encoder.positions[: table.shape[0]] = table
        matched, _ = encoder(mel, length)
        encoder.positions[: table.shape[0]] = torch.zeros_like(table)  # never reused

    generated_diff = float((generated - expected).abs().max())
    matched_diff = float((matched - expected).abs().max())
    scale = float(expected.abs().max())
    print(f"parity, generated positions: max |delta| = {generated_diff:.3e}")
    print(f"parity, positions matched:   max |delta| = {matched_diff:.3e} (gate {_PARITY_GATE})")
    if matched_diff > _PARITY_GATE and not args.skip_verify:
        raise SystemExit("parity gate FAILED — nothing written")

    state = encoder.state_dict()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state,
            "config": dataclasses.asdict(config),
            "source": args.model,
            "revision": revision,
            "parity": {
                "generated_max_abs": generated_diff,
                "matched_max_abs": matched_diff,
                "position_table_max_abs": report["position_max_diff"],
                "reference_scale": scale,
            },
            "sha256_state": state_sha256(state),
        },
        args.out,
    )
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(
        json.dumps(
            {"source": args.model, "revision": revision, "sha256_state": state_sha256(state)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
