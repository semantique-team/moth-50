"""Build the CS-FLEURS-READ IT-EN manifest (E-008, test-only).

CS-FLEURS (`byan/cs-fleurs`, CC-BY-NC-4.0) ships one metadata row per language
pair; this tool keeps the Italian-English read test (176 utterances, ~0.51 h,
one speaker) and writes it in the project manifest schema. It is **never
training data** and must stay `split: test`.

The transcript is normalized with the canonical `moth_text.normalize` rule
(`lang_hint` default `it`). 27 utterances contain digits and the normalizer
spells them in Italian; the report counts them, because an English-read number
is then a scoring artifact, not a recognition error. The manifest keeps all 176
rows.

    uv run --frozen -- python tools/build_cs_fleurs_manifest.py \\
      --out data/manifests/cs-fleurs-read.jsonl \\
      --report runs/E-008/cs-fleurs-read-report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "data" / "raw" / "cs-fleurs" / "read" / "test" / "metadata.jsonl"
DEFAULT_AUDIO_ROOT = ROOT / "data" / "raw" / "cs-fleurs" / "read" / "test"
LANGUAGE = "ita-eng"


def build_rows(metadata: Iterable[dict], language: str = LANGUAGE) -> list[dict]:
    """Manifest rows for one language pair, in metadata order."""
    sys.path.insert(0, str(ROOT / "vendor" / "moth-stt"))
    from moth_text import normalize

    rows = []
    for entry in metadata:
        if entry.get("language") != language:
            continue
        text = " ".join(entry["text"].split())
        rows.append(
            {
                "id": f"cs-fleurs-{entry['id']}",
                "audio": entry["file_name"],
                "text": text,
                "norm": normalize(text),
                "duration": float(entry["duration"]),
                "source": "cs-fleurs",
                "split": "test",
                "speaker": entry.get("speaker"),
                "lang": "mix",
                "category": None,
                "en_spans": [],
                "fluency": entry.get("fluency"),
            }
        )
    return rows


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--language", default=LANGUAGE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = build_rows(_jsonl(args.metadata), args.language)
    missing = [row["audio"] for row in rows if not (args.audio_root / row["audio"]).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} clips missing under {args.audio_root}: {missing[:3]}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate manifest ids")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    with_digits = sum(any(char.isdigit() for char in row["text"]) for row in rows)
    report = {
        "tool": "tools/build_cs_fleurs_manifest.py",
        "metadata": str(args.metadata),
        "metadata_sha256": _sha256(args.metadata),
        "audio_root": str(args.audio_root),
        "language": args.language,
        "out": str(args.out),
        "out_sha256": _sha256(args.out),
        "rows": len(rows),
        "hours": round(sum(row["duration"] for row in rows) / 3600, 4),
        "speakers": dict(Counter(row["speaker"] for row in rows)),
        "fluency": dict(Counter(row.get("fluency") for row in rows)),
        "rows_with_digits": with_digits,
        "digits_note": (
            "normalize(lang_hint='it') spells these in Italian; English-read numbers are a "
            "scoring artifact. All rows are kept; report the count."
        ),
        "split": "test",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
