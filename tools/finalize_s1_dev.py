"""Turn a ``build_manifest`` fragment into the S1-dev dev manifest.

``scripts/build_manifest.py`` writes the canonical ``audio/R-S1/...`` path and
``split: train``. The held-out S1-dev contract is the opposite: ``split: dev``
and ``audio/R-S1-dev/...``. This patches the fragment, verifies that every
audio path exists, and refuses to write anything that breaks the contract.

    cd ../moth-cs-dataset
    python -m scripts.build_manifest build --kind R --speaker S1 \
      --scripts-dir text-2026-11-dev --audio-dir audio/R-S1-dev \
      --out manifests/R-S1-dev.jsonl
    uv run --no-project python ../moth-50/tools/finalize_s1_dev.py \
      --fragment manifests/R-S1-dev.jsonl --dataset-root . --check

Run without ``--check`` to write in place (or to ``--out``). The transform is
idempotent: a second run reports "already finalized" and changes nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPEAKER = "S1"
SOURCE_PREFIX = "audio/R-S1/"
TARGET_PREFIX = "audio/R-S1-dev/"

MIN_SECONDS = 1.0
MAX_SECONDS = 30.0
BUDGET_MINUTES = (45.0, 60.0)


def load_rows(path: Path) -> list[dict]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def is_finalized(rows: list[dict]) -> bool:
    return all(
        row.get("split") == "dev" and str(row.get("audio", "")).startswith(TARGET_PREFIX)
        for row in rows
    )


def rewrite(rows: list[dict]) -> list[dict]:
    """Return dev rows pointing at the session audio; raise on a mixed fragment."""
    out: list[dict] = []
    for row in rows:
        audio = str(row.get("audio", ""))
        if audio.startswith(SOURCE_PREFIX):
            audio = TARGET_PREFIX + audio[len(SOURCE_PREFIX) :]
        elif not audio.startswith(TARGET_PREFIX):
            raise SystemExit(f"{row.get('id')}: unexpected audio path {audio!r}")
        out.append({**row, "split": "dev", "audio": audio})
    return out


def problems(rows: list[dict], root: Path) -> list[str]:
    """Contract violations: duplicates, missing audio, bad speaker/split/duration."""
    found: list[str] = []
    seen: set[str] = set()
    for row in rows:
        identifier = row.get("id")
        if not identifier:
            found.append("row without id")
        elif identifier in seen:
            found.append(f"{identifier}: duplicate id")
        seen.add(identifier)
        if row.get("speaker") != SPEAKER:
            found.append(f"{identifier}: speaker {row.get('speaker')!r}, expected {SPEAKER}")
        if row.get("split") != "dev":
            found.append(f"{identifier}: split {row.get('split')!r}, expected 'dev'")
        audio = row.get("audio")
        if not audio:
            found.append(f"{identifier}: missing audio")
        elif not (root / audio).is_file():
            found.append(f"{identifier}: audio not found: {root / audio}")
        duration = row.get("duration")
        if not isinstance(duration, (int, float)):
            found.append(f"{identifier}: missing duration")
        elif not MIN_SECONDS <= duration <= MAX_SECONDS:
            found.append(
                f"{identifier}: duration {duration}s outside {MIN_SECONDS:g}-{MAX_SECONDS:g}s"
            )
    return found


def summarize(rows: list[dict]) -> str:
    minutes = sum(float(row["duration"]) for row in rows) / 60
    mean = sum(float(row["duration"]) for row in rows) / len(rows)
    low, high = BUDGET_MINUTES
    verdict = "inside" if low <= minutes <= high else "outside"
    return (
        f"{len(rows)} rows, {minutes:.2f} min ({mean:.2f} s mean); "
        f"{verdict} the {low:g}-{high:g} min dev budget"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finalize_s1_dev.py", description=__doc__)
    parser.add_argument("--fragment", required=True, type=Path, help="build_manifest output")
    parser.add_argument(
        "--dataset-root", required=True, type=Path, help="root the audio paths are relative to"
    )
    parser.add_argument("--out", type=Path, help="write here (default: patch --fragment)")
    parser.add_argument("--check", action="store_true", help="report only, write nothing")
    args = parser.parse_args(argv)

    rows = load_rows(args.fragment)
    if is_finalized(rows):
        print(f"already finalized: {args.fragment}")
        print(f"  {summarize(rows)}")
        return 0

    rows = rewrite(rows)
    found = problems(rows, args.dataset_root)
    print(f"{args.fragment}: {summarize(rows)}")
    print(f"  {SOURCE_PREFIX} -> {TARGET_PREFIX}, split -> dev")
    if found:
        print(f"  FAILED with {len(found)} problem(s):")
        for item in found[:40]:
            print(f"    - {item}")
        return 1
    if args.check:
        print("  check ok; run without --check to write")
        return 0

    out = args.out or args.fragment
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"  wrote {out}")
    print("  next: build the mel shards (experiments/E-002-s1-dev.md §5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
