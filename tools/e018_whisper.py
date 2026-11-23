"""Whisper teacher runner for E-018.

Produces ``{id, hyp, latency_s}`` JSONL for the gold train set with a pinned
local Whisper checkpoint. Scoring is separate (Moth's ``stt/eval.py`` or the
vendored ``moth_eval``), so this tool never touches the scorer.

    uv run --frozen --project ../moth-stt --extra training -- \\
      python tools/e018_whisper.py \\
      --model-dir /path/to/whisper-large-v3-turbo \\
      --manifest ../moth-cs-dataset/manifests/train.jsonl \\
      --audio-root ../moth-cs-dataset --language it --device mps \\
      --out-dir runs/E-018/whisper-large-v3-turbo-it
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(__file__).resolve().parents[1]
MOTH = ROOT.parent / "moth-stt"
sys.path.insert(0, str(MOTH))

from stt.data.build_mel import load_audio_16k


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--language", default="it", choices=["it", "auto"])
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    processor = WhisperProcessor.from_pretrained(str(args.model_dir), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        str(args.model_dir), local_files_only=True, torch_dtype=torch.float32
    )
    model.eval()
    model.to(args.device)
    device = torch.device(args.device)

    started = time.time()
    audio_seconds = 0.0
    errors = []
    hyps = []
    for index, row in enumerate(rows, 1):
        try:
            pcm = load_audio_16k(args.audio_root / row["audio"])
            audio_seconds += len(pcm) / 16000
            inputs = processor(pcm, sampling_rate=16000, return_tensors="pt")
            features = inputs.input_features.to(device)
            generate_args = {"task": "transcribe", "max_new_tokens": 440}
            if args.language != "auto":
                generate_args["language"] = args.language
            clip_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(features, **generate_args)
            if device.type == "mps":
                torch.mps.synchronize()
            latency = time.perf_counter() - clip_started
            text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            hyps.append({"id": row["id"], "hyp": text, "latency_s": round(latency, 4)})
        except Exception as exc:  # noqa: BLE001 - per-clip error
            errors.append({"id": row.get("id"), "error": f"{type(exc).__name__}: {exc}"})
        if index % 50 == 0:
            print(f"{index}/{len(rows)} clips", flush=True)

    wall = time.time() - started
    out = args.out_dir / "gold_train.jsonl"
    out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in hyps), encoding="utf-8"
    )
    meta = {
        "tool": "tools/e018_whisper.py",
        "system": args.name or "whisper-large-v3-turbo",
        "model_dir": str(args.model_dir),
        "manifest": str(args.manifest),
        "audio_root": str(args.audio_root),
        "language": args.language,
        "device": args.device,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "clips": len(hyps),
        "errors": errors,
        "audio_seconds": round(audio_seconds, 2),
        "wall_seconds": round(wall, 2),
        "rtfx": round(audio_seconds / wall, 3) if wall else None,
        "output": str(out),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: meta[k]
                for k in ("system", "clips", "errors", "audio_seconds", "wall_seconds", "rtfx")
            },
            indent=2,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
