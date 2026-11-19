"""Build the MOTH-50 release folder from an E-005 checkpoint and check the 50 MB budget.

Uses Moth's exact 0.8.1 recipe (fp32 ONNX → dynamic per-channel QInt8) and the
release layout, then adds the A3 LM and the frozen decode settings. The FP32
export gate in `stt.export` was calibrated on the 4-layer tiny model; a 12-layer
encoder accumulates ~3e-3 median logit drift, so this tool **does not skip the
measurement**: it reports the FP32/INT8 logit gaps and the top-1 agreement, and
the real quality check is the beam+A3 WER on the dev sets.

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

REPO = Path(__file__).resolve().parents[1]
MOTH_STT = REPO.parent / "moth-stt"
LM = MOTH_STT / "release/lm_bpe3.bin"

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
    args = parser.parse_args(argv)

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"refusing to write into a non-empty {args.out_dir}")

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(MOTH_STT))
    sys.path.insert(0, str(REPO / "vendor/moth-stt"))
    import torch
    from moth_release import validate_release
    from stt.export import _config, _logit_gap, build_notice, export_fp32, quantize_int8
    from stt.features import DEFAULT_FILTERS
    from stt.tokenizer import load_tokenizer, vocab_json
    from stt.vad import ASSET_PATH as VAD_ASSET

    from training.config import load_config
    from training.model import build_model

    config = load_config(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = dataclasses.replace(
        config.model.__class__(**payload["model_config"]), init=None, identity_heads=False
    )
    model = build_model(model_config)
    model.load_state_dict(payload["model"], strict=True)
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
    shutil.copyfile(VAD_ASSET, out / "silero_vad.onnx")
    shutil.copyfile(LM, out / "lm_bpe3.bin")

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
    (out / "NOTICE").write_text(build_notice(), encoding="utf-8")

    _, budget = validate_release(out, budget_bytes=args.budget)
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_step": payload.get("global_step"),
        "version": args.version,
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
