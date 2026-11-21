"""Match one recorded set's level to a reference set with a fixed gain.

The feature path has no loudness normalisation (``stt.features.log_mel`` keeps a
per-clip floor and a fixed scale), so a recording made at a different input gain
reaches the encoder as a constant offset. This tool measures the median RMS of a
source directory and a reference directory, applies the **same** gain to every
source file (never per-file normalisation), and writes new PCM files. The raw
recordings are left untouched. A positive gain is refused unless explicitly
allowed, and the output is checked for clipping.

    cd ../moth-cs-dataset
    uv run --no-project --with numpy --with soundfile -- \
      python ../moth-50/tools/match_audio_level.py \
      --source audio/R-S1-cal/S1 --reference audio/R-S1-dev/S1 \
      --out audio/R-S1-cal-norm/S1 --report ../moth-50/runs/E-017/level-match.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

CLIP_LEVEL = 0.999


def wav_files(directory: Path) -> list[Path]:
    return sorted(Path(directory).glob("*.wav"))


def rms_dbfs(samples) -> float:
    import numpy as np

    array = np.asarray(samples, dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.square(array)))) if array.size else 0.0
    return 20.0 * math.log10(rms + 1e-12)


def peak(samples) -> float:
    import numpy as np

    array = np.asarray(samples, dtype=np.float64)
    return float(np.max(np.abs(array))) if array.size else 0.0


def read_mono(path: Path):
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return np.asarray(mono, dtype=np.float64), rate, data.shape[1]


def set_levels(directory: Path) -> dict:
    values = [rms_dbfs(read_mono(path)[0]) for path in wav_files(directory)]
    if not values:
        raise SystemExit(f"no WAV files in {directory}")
    values.sort()
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return {
        "files": len(values),
        "median_dbfs": median,
        "min_dbfs": values[0],
        "max_dbfs": values[-1],
    }


def gain_for(source: dict, reference: dict) -> float:
    return float(reference["median_dbfs"] - source["median_dbfs"])


def write_at_gain(source: Path, out: Path, gain_db: float) -> dict:
    import numpy as np
    import soundfile as sf

    factor = 10.0 ** (gain_db / 20.0)
    out.mkdir(parents=True, exist_ok=True)
    levels: list[float] = []
    peaks: list[float] = []
    clipped = 0
    for path in wav_files(source):
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        info = sf.info(str(path))
        scaled = np.asarray(data, dtype=np.float64) * factor
        target = out / path.name
        sf.write(str(target), scaled, rate, subtype=info.subtype)
        levels.append(rms_dbfs(scaled.mean(axis=1)))
        peaks.append(peak(scaled))
        clipped += int(np.sum(np.abs(scaled) >= CLIP_LEVEL))
    return {
        "files": len(levels),
        "median_dbfs": float(np.median(levels)) if levels else None,
        "peak_max": float(np.max(peaks)) if peaks else None,
        "clipped_samples": clipped,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="match_audio_level.py", description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="raw WAV directory")
    parser.add_argument("--reference", required=True, type=Path, help="target-level WAV directory")
    parser.add_argument("--out", required=True, type=Path, help="new WAV directory")
    parser.add_argument("--report", type=Path, help="write the measurements as JSON")
    parser.add_argument("--dry-run", action="store_true", help="measure only, write nothing")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output WAVs")
    parser.add_argument(
        "--allow-boost",
        action="store_true",
        help="allow a positive gain (checked for clipping after writing)",
    )
    args = parser.parse_args(argv)

    source = args.source.resolve()
    out = args.out.resolve()
    if source == out:
        raise SystemExit("--out must differ from --source: raw audio is never overwritten")
    if out.exists() and wav_files(out) and not args.overwrite:
        raise SystemExit(f"{out} already contains WAV files; pass --overwrite to replace them")

    source_levels = set_levels(source)
    reference_levels = set_levels(args.reference)
    gain_db = gain_for(source_levels, reference_levels)
    print(
        f"source {source}: {source_levels['files']} files, "
        f"median {source_levels['median_dbfs']:.2f} dBFS"
    )
    print(
        f"reference {args.reference}: {reference_levels['files']} files, "
        f"median {reference_levels['median_dbfs']:.2f} dBFS"
    )
    print(f"gain: {gain_db:+.2f} dB (fixed, not per-file)")
    if gain_db > 0 and not args.allow_boost:
        raise SystemExit("refusing a positive gain without --allow-boost")
    if args.dry_run:
        print("dry run: nothing written")
        return 0

    result = write_at_gain(source, out, gain_db)
    print(
        f"wrote {result['files']} files to {out}: median {result['median_dbfs']:.2f} dBFS, "
        f"peak max {result['peak_max']:.3f}, clipped samples {result['clipped_samples']}"
    )
    report = {
        "tool": "tools/match_audio_level.py",
        "source_dir": str(args.source),
        "reference_dir": str(args.reference),
        "out_dir": str(args.out),
        "gain_db": round(gain_db, 3),
        "source": source_levels,
        "reference": reference_levels,
        "out_levels": result,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.report}")
    if result["clipped_samples"]:
        print(f"FAILED: {result['clipped_samples']} clipped samples after gain")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
