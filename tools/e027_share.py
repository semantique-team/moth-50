"""E-027 TTS share: the repeat that makes the TTS audio about 10 % of an epoch.

    uv run --frozen --project ../moth-stt --extra training -- \\
      python tools/e027_share.py --config training/configs/base6-e027-control.toml \\
      --tts-manifest data/manifests/tts_e027.jsonl --out runs/E-027/share.json

``repeat = max(1, round(share / (1 - share) * base_hours / tts_hours))``, where
the base hours are the control mixture with its own repeats. The TTS arm's
``repeat`` is set to this value and committed before the first training step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def manifest_hours(path: Path) -> float:
    seconds = 0.0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            seconds += float(json.loads(line)["duration"])
    return seconds / 3600


def tts_repeat(base_hours: float, tts_hours: float, share: float = 0.10) -> int:
    if not 0.0 < share < 1.0:
        raise ValueError("share must be in (0, 1)")
    if base_hours <= 0 or tts_hours <= 0:
        raise ValueError("hours must be positive")
    return max(1, round(share / (1.0 - share) * base_hours / tts_hours))


def main(argv: list[str] | None = None) -> int:
    from training.config import load_config

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="the control config")
    parser.add_argument("--tts-manifest", type=Path, required=True)
    parser.add_argument("--share", type=float, default=0.10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    sources = []
    base_hours = 0.0
    for source in config.data.train:
        hours = manifest_hours(source.manifest)
        base_hours += hours * source.repeat
        sources.append(
            {"manifest": str(source.manifest), "repeat": source.repeat, "hours": round(hours, 3)}
        )
    tts_hours = manifest_hours(args.tts_manifest)
    repeat = tts_repeat(base_hours, tts_hours, args.share)
    report = {
        "tool": "tools/e027_share.py",
        "config": str(args.config),
        "sources": sources,
        "base_hours": round(base_hours, 3),
        "tts_manifest": str(args.tts_manifest),
        "tts_hours": round(tts_hours, 3),
        "target_share": args.share,
        "repeat": repeat,
        "actual_share": round(tts_hours * repeat / (base_hours + tts_hours * repeat), 4),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
