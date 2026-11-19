"""E-012 probe: is the 16 kHz-only Silero graph a drop-in for the shipped one?

The shipped ``silero_vad.onnx`` (2,327,524 B) holds an 8 kHz and a 16 kHz branch
behind an ``If``; MOTH-50 only ever feeds 16 kHz. The same Silero commit ships
``silero_vad_16k_op15.onnx`` (1,289,603 B) with the same inputs and outputs
(``input``, ``state``, ``sr`` → ``output``, ``stateN``). If both graphs give the same
window probabilities, clip gate and segments on real audio, the release saves
1,037,921 B without touching the runtime.

The probe runs Moth's own VAD path (``stt.vad``: 512 + 64-sample windows, state
carry, 0.5 window threshold, 0.10 clip gate, ``segment_bounds``) with both graphs:

    python tools/release_assets.py fetch vad vad-16k
    uv run --frozen --project ../moth-stt --extra training -- \
      python tools/probe_vad.py --moth-stt ../moth-stt \
      --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl --audio-root ../moth-cs-dataset \
      --manifest ../moth-cs-dataset/manifests/test.jsonl --audio-root ../moth-cs-dataset \
      --manifest data/manifests/noise.jsonl --audio-root data/raw/noise \
      --out runs/E-012/vad-parity.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from release_assets import SILERO_VAD, SILERO_VAD_16K, sha256, verify

AUDIO_SUFFIXES = {".wav", ".flac"}
#: Largest window-probability gap still called identical (float32 graph rewrites).
TOLERANCE = 1e-4


def collect(manifests: list[Path], roots: list[Path], audio: list[Path]) -> list[tuple[str, Path]]:
    """``(id, path)`` for every manifest row and every audio file under ``audio``."""
    clips: list[tuple[str, Path]] = []
    for manifest, root in zip(manifests, roots, strict=True):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                clips.append((f"{manifest.stem}:{row['id']}", root / row["audio"]))
    for entry in audio:
        files = sorted(entry.rglob("*")) if entry.is_dir() else [entry]
        clips.extend(
            (f"file:{path.name}", path) for path in files if path.suffix.lower() in AUDIO_SUFFIXES
        )
    return clips


def compare(reference, candidate, *, window_threshold: float, gate: float) -> dict:
    """Per-clip agreement of two window-probability arrays."""
    import numpy as np

    gap = np.abs(reference - candidate)
    ref_voiced = reference > window_threshold
    cand_voiced = candidate > window_threshold
    ref_fraction = float(ref_voiced.mean()) if reference.size else 0.0
    cand_fraction = float(cand_voiced.mean()) if candidate.size else 0.0
    return {
        "windows": int(reference.size),
        "max_abs": float(gap.max()) if gap.size else 0.0,
        "mean_abs": float(gap.mean()) if gap.size else 0.0,
        "window_flips": int((ref_voiced != cand_voiced).sum()),
        "gate_reference": ref_fraction > gate,
        "gate_candidate": cand_fraction > gate,
    }


def verdict(results: list[dict]) -> dict:
    max_abs = max((result["max_abs"] for result in results), default=0.0)
    gate_flips = [
        result["id"] for result in results if result["gate_reference"] != result["gate_candidate"]
    ]
    segment_flips = [result["id"] for result in results if not result["segments_equal"]]
    window_flips = sum(result["window_flips"] for result in results)
    return {
        "clips": len(results),
        "max_abs": max_abs,
        "window_flips": window_flips,
        "gate_disagreements": gate_flips,
        "segment_disagreements": segment_flips,
        "drop_in": max_abs <= TOLERANCE and not gate_flips and not segment_flips,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="probe_vad.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--moth-stt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--audio-root", type=Path, action="append", default=[])
    parser.add_argument("--audio", type=Path, action="append", default=[], help="file or directory")
    parser.add_argument("--reference", type=Path, default=SILERO_VAD.path)
    parser.add_argument("--candidate", type=Path, default=SILERO_VAD_16K.path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.manifest) != len(args.audio_root):
        parser.error("give one --audio-root per --manifest")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if args.reference == SILERO_VAD.path:
        verify(SILERO_VAD)
    if args.candidate == SILERO_VAD_16K.path:
        verify(SILERO_VAD_16K)

    sys.path.insert(0, str(args.moth_stt.resolve()))
    import onnxruntime as ort
    from stt.data.build_mel import load_audio_16k
    from stt.vad import GATE_THRESHOLD, WINDOW_THRESHOLD, segment_bounds, speech_probabilities

    sessions = {
        name: ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for name, path in (("reference", args.reference), ("candidate", args.candidate))
    }
    clips = collect(args.manifest, args.audio_root, args.audio)
    if not clips:
        parser.error("no audio given")
    seconds = {"reference": 0.0, "candidate": 0.0}
    audio_seconds = 0.0
    results = []
    for uid, path in clips:
        audio = load_audio_16k(path)
        audio_seconds += audio.size / 16000
        probabilities = {}
        for name, session in sessions.items():
            started = time.perf_counter()
            probabilities[name] = speech_probabilities(audio, session=session)
            seconds[name] += time.perf_counter() - started
        result = {"id": uid}
        result |= compare(
            probabilities["reference"],
            probabilities["candidate"],
            window_threshold=WINDOW_THRESHOLD,
            gate=GATE_THRESHOLD,
        )
        result["segments_equal"] = segment_bounds(
            audio, probabilities=probabilities["reference"]
        ) == segment_bounds(audio, probabilities=probabilities["candidate"])
        results.append(result)

    summary = verdict(results)
    output = {
        "tool": "tools/probe_vad.py",
        "graphs": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in (("reference", args.reference), ("candidate", args.candidate))
        },
        "saved_bytes": args.reference.stat().st_size - args.candidate.stat().st_size,
        "audio_seconds": round(audio_seconds, 2),
        "vad_seconds": {name: round(value, 2) for name, value in seconds.items()},
        "tolerance": TOLERANCE,
        "summary": summary,
        "clips": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("saved_bytes", "audio_seconds", "vad_seconds")}))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
