"""Build the MOTH-50 release folder from an E-005 checkpoint and check the 50 MB budget.

Uses Moth's exact 0.8.1 recipe (fp32 ONNX → dynamic per-channel QInt8) and the
release layout, then adds the 0.8.1 A3 LM (pinned and sha256-checked by
`tools/release_assets.py`) and the frozen decode settings. The FP32
export gate in `stt.export` was calibrated on the 4-layer tiny model; a 12-layer
encoder accumulates ~3e-3 median logit drift, so this tool **does not skip the
measurement**: it reports the FP32/INT8 logit gaps and the top-1 agreement, and
the real quality check is the beam+A3 WER on the dev sets.

    python tools/release_assets.py fetch lm
    uv run --frozen --project ../moth-stt --extra export -- \
      python tools/export_release.py \
      --checkpoint runs/E-005/base+6-16k/checkpoints/ckpt_best.pt \
      --config training/configs/base6-long.toml \
      --out-dir runs/E-005/release --version 0.9.0-e005
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
from pathlib import Path

from moth_lm import read_mothlmb1, write_mothlmb2
from release_assets import SILERO_VAD, SILERO_VAD_16K, resolve_lm, verify
from release_notice import compose_notice, moth_dataset_lines

REPO = Path(__file__).resolve().parents[1]
MOTH_STT = REPO.parent / "moth-stt"

#: ``--vad`` choices: the asset and the NOTICE wording. The file is always
#: shipped as ``silero_vad.onnx`` so the runtime contract does not change.
VAD_VARIANTS = {
    "full": (SILERO_VAD, "the 8 kHz + 16 kHz graph shipped by Moth 0.8.1"),
    "16k": (SILERO_VAD_16K, "the 16 kHz-only graph silero_vad_16k_op15.onnx"),
}

FILES = (
    "model.int8.onnx",
    "vocab.json",
    "mel_filters.bin",
    "silero_vad.onnx",
    "lm_bpe3.bin",
)


def sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="export_release.py", description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--version", default="0.9.0-e005")
    parser.add_argument("--budget", type=int, default=50_000_000)
    parser.add_argument(
        "--lm", type=Path, default=None, help="decoder LM (default: the pinned 0.8.1 A3 LM)"
    )
    parser.add_argument(
        "--allow-other-lm",
        action="store_true",
        help="package an LM that is not A3 (E-004 LM arms only)",
    )
    parser.add_argument(
        "--weights",
        default="auto",
        choices=["auto", "model", "ema"],
        help="checkpoint weights: auto = EMA when present (training/ema.py)",
    )
    parser.add_argument(
        "--vad",
        default="16k",
        choices=sorted(VAD_VARIANTS),
        help="Silero graph: 16k (default since D19, -1.04 MB) or full (8+16 kHz, as in 0.8.1)",
    )
    parser.add_argument(
        "--lm-format",
        default="mothlmb2",
        choices=("mothlmb1", "mothlmb2"),
        help="LM encoding: mothlmb2 is the lossless packed format (default since D20)",
    )
    args = parser.parse_args(argv)

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"refusing to write into a non-empty {args.out_dir}")
    lm_info = resolve_lm(args.lm, allow_other=args.allow_other_lm)
    vad_asset, vad_description = VAD_VARIANTS[args.vad]

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(MOTH_STT))
    sys.path.insert(0, str(REPO / "vendor/moth-stt"))
    import torch
    from moth_release import validate_release
    from stt.data.sources import SOURCES
    from stt.export import (
        _SILERO_MIT,
        _WHISPER_MIT,
        _config,
        _logit_gap,
        export_fp32,
        quantize_int8,
    )
    from stt.features import DEFAULT_FILTERS
    from stt.tokenizer import load_tokenizer, vocab_json
    from stt.vad import ASSET_PATH as VAD_ASSET

    from training.config import load_config
    from training.ema import checkpoint_weights
    from training.model import build_model

    config = load_config(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = dataclasses.replace(
        config.model.__class__(**payload["model_config"]), init=None, identity_heads=False
    )
    model = build_model(model_config)
    state, weights = checkpoint_weights(payload, args.weights)
    model.load_state_dict(state, strict=True)
    model = model.cpu().eval()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    fp32_path = out / "model.fp32.onnx"
    export_fp32(model, fp32_path)
    fp32_gap = _logit_gap(model, fp32_path, (1.0, 5.0, 30.0), seed=0)
    int8_path = out / "model.int8.onnx"
    quantize_int8(fp32_path, int8_path)
    int8_gap = _logit_gap(model, int8_path, (1.0, 5.0), seed=0)
    fp32_path.unlink(missing_ok=True)

    tokenizer = load_tokenizer(config.data.tokenizer)
    pieces = {int(key): piece for key, piece in vocab_json(tokenizer).items()}
    (out / "vocab.json").write_text(
        json.dumps(
            {str(identifier): pieces[identifier] for identifier in sorted(pieces)},
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(DEFAULT_FILTERS, out / "mel_filters.bin")
    vad_path = verify(vad_asset, VAD_ASSET if args.vad == "full" else None)
    shutil.copyfile(vad_path, out / "silero_vad.onnx")
    if args.lm_format == "mothlmb2":
        lm_bytes = write_mothlmb2(read_mothlmb1(lm_info["path"].read_bytes()), bits=10)
        (out / "lm_bpe3.bin").write_bytes(lm_bytes)
        lm_packed = {
            "format": "MOTHLMB2",
            "bits": 10,
            "bytes": len(lm_bytes),
            "sha256": sha256(out / "lm_bpe3.bin"),
        }
    else:
        shutil.copyfile(lm_info["path"], out / "lm_bpe3.bin")
        lm_packed = {
            "format": "MOTHLMB1",
            "bits": None,
            "bytes": (out / "lm_bpe3.bin").stat().st_size,
            "sha256": sha256(out / "lm_bpe3.bin"),
        }

    release_config = _config(args.version)
    # The web runtime's release contract checks this name; MOTH-50 is the model
    # and display name, while the runtime/package lineage stays ``moth-stt``.
    release_config["name"] = "moth-stt"
    release_config["budget_bytes"] = args.budget
    release_config["decode"] = {
        "default": "beam",
        "beam": 12,
        "top_k": 10,
        "lm": "lm_bpe3.bin",
        "lm_weight": 0.55,
        "token_bonus": 2.7,
    }
    release_config["files"] = {name: f"sha256:{sha256(out / name)}" for name in FILES}
    (out / "config.json").write_text(
        json.dumps(release_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    notice = compose_notice(
        whisper_mit=_WHISPER_MIT,
        silero_mit=_SILERO_MIT,
        lock=json.loads((REPO / "data/sources.lock.json").read_text(encoding="utf-8")),
        moth_lines=moth_dataset_lines(SOURCES.values()),
        encoder_preset=model_config.preset,
        head_blocks=model_config.head_blocks,
        lm_sha256=lm_info["sha256"],
        vad_description=vad_description,
    )
    (out / "NOTICE").write_text(notice, encoding="utf-8")

    _, budget = validate_release(out, budget_bytes=args.budget)
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_step": payload.get("global_step"),
        "checkpoint_weights": weights,
        "version": args.version,
        "lm": {"sha256": lm_info["sha256"], "label": lm_info["label"], "is_a3": lm_info["is_a3"]},
        "lm_packed": lm_packed,
        "vad": {"variant": args.vad, "sha256": vad_asset.sha256},
        "gaps": {"fp32": fp32_gap, "int8": int8_gap},
        "budget": budget,
        "files": {name: budget["files"][name]["bytes"] for name in sorted(budget["files"])},
    }
    (out / "export-report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "fp32_gap": {
                    "max_abs": fp32_gap["max_abs"],
                    "top1_agreement": fp32_gap["top1_agreement"],
                },
                "int8_gap": {
                    "max_abs": int8_gap["max_abs"],
                    "top1_agreement": int8_gap["top1_agreement"],
                },
                "total_mb": round(budget["total_bytes"] / 1e6, 3),
                "budget_mb": round(budget["budget_bytes"] / 1e6, 3),
                "remaining_mb": round(budget["remaining_bytes"] / 1e6, 3),
                "files": result["files"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
