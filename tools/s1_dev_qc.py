"""S1-dev recording QC: coverage, levels and the remaining session size.

Measures whatever takes exist so far against the reviewed cue file, so a
session can be checked at any point instead of only at the end. Read-only: it
never writes audio, manifests or shards.

    cd ../moth-cs-dataset
    uv run --no-project --with numpy --with soundfile -- \
      python ../moth-50/tools/s1_dev_qc.py \
      --scripts text-2026-11-dev/R-S1.jsonl --audio audio/R-S1-dev/S1

Hard errors mirror ``scripts/build_manifest.py`` (1-30 s, <=1 % clipped, not
silent) and make the command exit 1. Warnings are advisory and never block the
manifest build.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

# Mirrors scripts/build_manifest.py: the manifest build rejects these.
MIN_SECONDS = 1.0
MAX_SECONDS = 30.0
SILENCE_PEAK = 1e-3
CLIPPING_RATIO = 0.01

# Advisory only.
HEADROOM_DBFS = -3.0
QUIET_DBFS = -24.0
LEVEL_DRIFT_DB = 3.0
LONG_SECONDS = 8.0
SHORT_SECONDS = 1.5

CUE_RE = re.compile(r"_(\d{4})\.wav$")


def cue_index(name: str) -> int:
    """1-based cue index of ``S1_0007.wav``; 0 when the name does not match."""
    match = CUE_RE.search(name)
    return int(match.group(1)) if match else 0


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def load_cues(path: Path | str) -> list[dict]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"no cues in {path}")
    return rows


def measure(path: Path) -> dict:
    """Duration, peak/RMS in dBFS and clipped-sample ratio for one take."""
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
    clipped = int(np.sum(np.abs(mono) >= 0.999))
    return {
        "name": path.name,
        "index": cue_index(path.name),
        "duration": round(mono.size / rate, 3),
        "peak_dbfs": round(20 * np.log10(peak + 1e-12), 1),
        "rms_dbfs": round(20 * np.log10(rms + 1e-12), 1),
        "clipping": round(clipped / max(mono.size, 1), 5),
    }


def qc(takes: list[dict], cues: list[dict]) -> dict:
    """Build the report from measured takes and the reviewed cue list."""
    takes = sorted(takes, key=lambda take: take["index"])
    expected = len(cues)
    indices = {take["index"] for take in takes}
    last = min(max(indices, default=0), expected)
    gaps = [index for index in range(1, last + 1) if index not in indices]
    remaining = expected - len(indices) - len(gaps)
    extra = sorted(index for index in indices if index <= 0 or index > expected)

    durations = [take["duration"] for take in takes]
    peaks = [take["peak_dbfs"] for take in takes]
    rms = [take["rms_dbfs"] for take in takes]

    errors: list[str] = []
    warnings: list[str] = []
    for take in takes:
        if not MIN_SECONDS <= take["duration"] <= MAX_SECONDS:
            errors.append(
                f"{take['name']}: {take['duration']:.2f}s outside {MIN_SECONDS:g}-{MAX_SECONDS:g}s"
            )
        elif take["duration"] < SHORT_SECONDS:
            warnings.append(f"{take['name']}: very short ({take['duration']:.2f}s)")
        if take["clipping"] > CLIPPING_RATIO:
            errors.append(f"{take['name']}: clipped on {take['clipping']:.1%} of samples")
        if take["peak_dbfs"] < 20 * math.log10(SILENCE_PEAK):
            errors.append(f"{take['name']}: silent (peak {take['peak_dbfs']:.1f} dBFS)")
        if take["peak_dbfs"] > HEADROOM_DBFS:
            warnings.append(f"{take['name']}: peak {take['peak_dbfs']:.1f} dBFS, little headroom")
        if take["peak_dbfs"] < QUIET_DBFS:
            warnings.append(f"{take['name']}: peak {take['peak_dbfs']:.1f} dBFS, quieter than S1")
    if gaps:
        warnings.append(f"{len(gaps)} cue(s) skipped inside the session: {gaps[:12]}")
    if extra:
        warnings.append(f"{len(extra)} take(s) beyond the cue list: {extra[:12]}")

    # Level drift across the session: a gain change mid-session is worth seeing.
    drift = None
    if len(takes) >= 20:
        half = len(takes) // 2
        first = sum(peaks[:half]) / half
        second = sum(peaks[half:]) / (len(takes) - half)
        drift = round(second - first, 1)
        if abs(drift) >= LEVEL_DRIFT_DB:
            warnings.append(f"level drift {drift:+.1f} dB between the first and second half")

    # Cue statistics and the projected full-session size, from the measured
    # seconds-per-word of the takes recorded so far.
    counts = [word_count(cue["text"]) for cue in cues]
    paired = [
        (word_count(cues[take["index"] - 1]["text"]), take["duration"])
        for take in takes
        if 1 <= take["index"] <= expected
    ]
    paired_words = sum(words for words, _ in paired)
    seconds_per_word = (
        sum(duration for _, duration in paired) / paired_words if paired_words else 0.0
    )
    estimated = [seconds_per_word * count for count in counts]

    categories: dict[str, int] = {}
    for take in takes:
        if 1 <= take["index"] <= expected:
            category = cues[take["index"] - 1].get("category", "?")
            categories[category] = categories.get(category, 0) + 1

    recorded_seconds = sum(durations)
    has_long = sum(1 for est in estimated if est > LONG_SECONDS)
    if takes and not has_long:
        warnings.append(f"no cue estimates over {LONG_SECONDS:g}s: the session is short-form only")

    return {
        "clips": len(takes),
        "expected": expected,
        "gaps": gaps,
        "remaining": remaining,
        "extra": extra,
        "errors": errors,
        "warnings": warnings,
        "audio": {
            "seconds": round(recorded_seconds, 1),
            "minutes": round(recorded_seconds / 60, 2),
            "mean_s": round(sum(durations) / len(durations), 2) if durations else 0.0,
            "median_s": round(sorted(durations)[len(durations) // 2], 2) if durations else 0.0,
            "min_s": round(min(durations), 2) if durations else 0.0,
            "max_s": round(max(durations), 2) if durations else 0.0,
        },
        "levels": {
            "peak_mean_dbfs": round(sum(peaks) / len(peaks), 1) if peaks else None,
            "peak_min_dbfs": min(peaks) if peaks else None,
            "peak_max_dbfs": max(peaks) if peaks else None,
            "rms_mean_dbfs": round(sum(rms) / len(rms), 1) if rms else None,
            "drift_db": drift,
        },
        "script": {
            "words": sum(counts),
            "mean_words": round(sum(counts) / len(counts), 1),
            "max_words": max(counts),
            "seconds_per_word": round(seconds_per_word, 3),
            "estimated_minutes": round(sum(estimated) / 60, 1),
            "estimated_remaining_minutes": round((sum(estimated) - recorded_seconds) / 60, 1),
            "estimated_over_8s": has_long,
            "categories": categories,
        },
    }


def render(report: dict) -> str:
    audio = report["audio"]
    levels = report["levels"]
    script = report["script"]
    percent = 100 * report["clips"] / report["expected"] if report["expected"] else 0.0
    lines = [
        (
            f"S1-dev QC: {report['clips']}/{report['expected']} takes ({percent:.1f} %), "
            f"{report['remaining']} still to record"
        ),
        (
            f"  audio   {audio['minutes']:.2f} min recorded, mean {audio['mean_s']:.2f} s, "
            f"range {audio['min_s']:.2f}-{audio['max_s']:.2f} s"
        ),
        (
            f"  levels  peak {levels['peak_mean_dbfs']} dBFS mean "
            f"({levels['peak_min_dbfs']}..{levels['peak_max_dbfs']}), "
            f"RMS {levels['rms_mean_dbfs']} dBFS, drift {levels['drift_db']} dB"
        ),
        (
            f"  script  {script['mean_words']} words mean (max {script['max_words']}), "
            f"~{script['estimated_minutes']} min total, "
            f"~{script['estimated_remaining_minutes']} min to go, "
            f"{script['estimated_over_8s']} cues over 8 s"
        ),
        "  cats    " + ", ".join(f"{k} {v}" for k, v in sorted(script["categories"].items())),
    ]
    if report["errors"]:
        lines.append(f"  ERRORS  {len(report['errors'])}")
        lines.extend(f"    - {item}" for item in report["errors"][:20])
    if report["warnings"]:
        lines.append(f"  warnings {len(report['warnings'])}")
        lines.extend(f"    - {item}" for item in report["warnings"][:20])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="s1_dev_qc.py", description=__doc__)
    parser.add_argument("--scripts", required=True, type=Path, help="reviewed cue file (JSONL)")
    parser.add_argument("--audio", required=True, type=Path, help="directory of S1_*.wav takes")
    parser.add_argument("--json", type=Path, help="write the full report (with per-take rows)")
    args = parser.parse_args(argv)

    cues = load_cues(args.scripts)
    takes = [measure(path) for path in sorted(args.audio.glob("*.wav"))]
    report = qc(takes, cues)
    print(render(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({**report, "takes": takes}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
