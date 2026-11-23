"""Combine the clean E-034c corpus with filtered E-020 generated text.

Writes a corpus JSONL whose rows all have a ``text`` field for
``tools/build_lm_from_corpus.py`` and reports exact-duplicate drops.

    uv run --frozen --project ../moth-stt --extra training -- \\
      python tools/e020_combine.py --base /tmp/lm-corpus-nos1dev.jsonl \\
      --generated runs/E-025/filtered.jsonl --out /tmp/lm-corpus-e025.jsonl \\
      --report runs/E-025/combine-report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOTH = ROOT.parent / "moth-stt"
sys.path.insert(0, str(MOTH))

from stt.text import normalize


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    seen = set()
    out_lines = []
    for line in args.base.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        seen.add(normalize(row["text"]))
        out_lines.append(line)

    added = 0
    duplicates = 0
    categories = Counter()
    for line in args.generated.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = normalize(row["text"])
        if text in seen:
            duplicates += 1
            continue
        seen.add(text)
        out_lines.append(json.dumps({"text": row["text"]}, ensure_ascii=False))
        categories[row.get("category", "?")] += 1
        added += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    report = {
        "tool": "tools/e020_combine.py",
        "base": {"path": str(args.base), "sha256": _sha(args.base), "rows": len(out_lines) - added},
        "generated": {"path": str(args.generated), "sha256": _sha(args.generated)},
        "added": added,
        "duplicate_rows_dropped": duplicates,
        "added_by_category": dict(categories),
        "total_rows": len(out_lines),
        "out": {"path": str(args.out), "sha256": _sha(args.out)},
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
