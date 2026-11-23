"""Canary / Voxtral teacher runners for E-018.

Both are inference-only and produce ``{id, hyp, latency_s}`` JSONL for the
gold train set. The scorer is separate (Moth's ``stt/eval.py``).

    uv run --frozen --project ../moth-stt --extra training -- \\
      python tools/e018_teacher.py --family canary \\
      --model-dir /path/to/canary-1b-v2 \\
      --manifest ../moth-cs-dataset/manifests/train.jsonl \\
      --audio-root ../moth-cs-dataset --language it --device mps \\
      --out-dir runs/E-018/canary-1b-v2-it

Voxtral needs ``mistral-common[audio]``; add
``--with "mistral-common[audio]"`` to the ``uv run`` command.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    CanaryForConditionalGeneration,
    VoxtralForConditionalGeneration,
)

ROOT = Path(__file__).resolve().parents[1]
MOTH = ROOT.parent / "moth-stt"
sys.path.insert(0, str(MOTH))

from stt.data.build_mel import load_audio_16k

MODELS = {
    "canary": CanaryForConditionalGeneration,
    "voxtral": VoxtralForConditionalGeneration,
}


def transcribe_canary(processor, model, pcm, device, language):
    inputs = processor.apply_transcription_request(
        audio=pcm,
        source_language=language,
        target_language=language,
        sampling_rate=16000,
    )
    inputs = inputs.to(device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=256)
    if device.type == "mps":
        torch.mps.synchronize()
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def transcribe_voxtral(processor, model, pcm, device, language, model_id):
    inputs = processor.apply_transcription_request(
        audio=pcm,
        model_id=model_id,
        language=language,
        sampling_rate=16000,
        format="wav",
    )
    inputs = inputs.to(device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=256)
    if device.type == "mps":
        torch.mps.synchronize()
    prompt_length = inputs.input_ids.shape[1]
    return processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)[0].strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", required=True, choices=sorted(MODELS))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--language", default="it")
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

    processor = AutoProcessor.from_pretrained(str(args.model_dir), local_files_only=True)
    model_cls = MODELS[args.family]
    dtype = torch.bfloat16 if args.family == "voxtral" else torch.float32
    try:
        model = model_cls.from_pretrained(
            str(args.model_dir), local_files_only=True, torch_dtype=dtype
        )
    except (TypeError, ValueError):
        model = model_cls.from_pretrained(str(args.model_dir), local_files_only=True)
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
            clip_started = time.perf_counter()
            if args.family == "canary":
                text = transcribe_canary(processor, model, pcm, device, args.language)
            else:
                text = transcribe_voxtral(
                    processor, model, pcm, device, args.language, str(args.model_dir)
                )
            latency = time.perf_counter() - clip_started
            hyps.append({"id": row["id"], "hyp": text, "latency_s": round(latency, 4)})
        except Exception as exc:  # noqa: BLE001 - per-clip error
            errors.append({"id": row.get("id"), "error": f"{type(exc).__name__}: {exc}"})
        if index % 25 == 0:
            print(f"{index}/{len(rows)} clips", flush=True)

    wall = time.time() - started
    out = args.out_dir / "gold_train.jsonl"
    out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in hyps), encoding="utf-8"
    )
    meta = {
        "tool": "tools/e018_teacher.py",
        "system": args.name or args.family,
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
