"""Build the E-017 R-S1-cal script: spans, schema, exact/near dedupe.

Candidates are Python modules with ``ROWS = [(category, text, [terms...]), ...]``.
The tool computes ``en_spans`` from the terms, validates every row with the
dataset's own gatekeeper (``scripts.generate_scripts.validate_candidate``),
rejects exact and near duplicates (3-gram Jaccard > 0.8) against every script
given with ``--scripts`` (and against the candidates themselves), and rejects
any sentence that is already an exact sentence of the LM corpus.

    python tools/build_cal_script.py \\
      --candidates runs/E-017/candidates_a.py \\
      --candidates runs/E-017/candidates_b.py \\
      --scripts ../moth-cs-dataset/text/T.jsonl \\
      --scripts ../moth-cs-dataset/text/A.jsonl \\
      --scripts ../moth-cs-dataset/text/R-S1.jsonl \\
      --scripts ../moth-cs-dataset/text/R-S2.jsonl \\
      --scripts ../moth-cs-dataset/text/pool.jsonl \\
      --scripts ../moth-cs-dataset/text-2026-11-dev/R-S1.jsonl \\
      --lm-corpus ../moth-stt/runs/E-031/frozen/lm-corpus.jsonl \\
      --out ../moth-cs-dataset/text-2026-11-cal/R-S1.jsonl \\
      --report runs/E-017/dedupe-report.json

No output is written if any candidate is invalid or duplicated.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO.parent / "moth-cs-dataset"
sys.path.insert(0, str(DATASET))

from scripts.generate_scripts import (
    _DedupeIndex,
    load_script,
    validate_candidate,
)
from scripts.text import normalize


def spans_for(text: str, terms: list[str]) -> list[list[int]]:
    tokens = text.split()
    spans = []
    used: set[int] = set()
    for term in terms:
        wanted = term.split()
        if not wanted:
            continue
        for start in range(len(tokens) - len(wanted) + 1):
            end = start + len(wanted)
            if tokens[start:end] == wanted and not (used & set(range(start, end))):
                spans.append([start, end])
                used.update(range(start, end))
                break
        else:
            raise ValueError(f"term {term!r} not found in {text!r}")
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--candidates", type=Path, action="append", required=True)
    parser.add_argument(
        "--scripts",
        type=Path,
        action="append",
        required=True,
        help="existing script JSONL to dedupe against",
    )
    parser.add_argument("--lm-corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--generator", default="deepseek-v4.1-flash")
    parser.add_argument("--date", default="2026-11-21")
    parser.add_argument("--prefix", default="R-S1-CAL")
    args = parser.parse_args()

    rows = []
    for path in args.candidates:
        rows.extend(runpy.run_path(str(path))["ROWS"])

    index = _DedupeIndex()
    existing_rows = 0
    for path in args.scripts:
        script = load_script(path)
        existing_rows += len(script)
        for row in script:
            index.add(row["text"])

    corpus_exact = set()
    for line in args.lm_corpus.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        corpus_exact.add(
            normalize(row["text"], "it" if row.get("lang") == "mix" else row.get("lang", "it"))
        )

    errors: list[str] = []
    duplicates: list[dict] = []
    corpus_hits: list[dict] = []
    built: list[dict] = []
    for category, text, terms in rows:
        try:
            spans = spans_for(text, terms)
            base = validate_candidate(
                {
                    "text": text,
                    "category": category,
                    "en_spans": spans,
                    "generator": args.generator,
                    "date": args.date,
                }
            )
        except Exception as exc:  # noqa: BLE001 - the report wants every failure
            errors.append(f"{category}: {text!r}: {exc}")
            continue
        if index.is_duplicate(text):
            duplicates.append({"category": category, "text": text})
            continue
        if normalize(text) in corpus_exact:
            corpus_hits.append({"category": category, "text": text})
            continue
        index.add(text)
        built.append(base)

    by_category = Counter(row["category"] for row in built)
    words = [len(row["text"].split()) for row in built]
    report = {
        "tool": "tools/build_cal_script.py",
        "candidates": [str(p) for p in args.candidates],
        "against": [str(p) for p in args.scripts],
        "lm_corpus": str(args.lm_corpus),
        "existing_rows": existing_rows,
        "candidates_rows": len(rows),
        "built_rows": len(built),
        "by_category": dict(sorted(by_category.items())),
        "words": {
            "min": min(words) if words else None,
            "max": max(words) if words else None,
            "mean": round(sum(words) / len(words), 2) if words else None,
        },
        "invalid": errors,
        "duplicates": duplicates,
        "lm_corpus_exact": corpus_hits,
    }

    if errors or duplicates or corpus_hits:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            f"REFUSED: {len(errors)} invalid, {len(duplicates)} duplicate, "
            f"{len(corpus_hits)} LM-corpus exact; see {args.report}"
        )
        for item in errors[:20]:
            print("  invalid:", item)
        for item in duplicates[:20]:
            print("  duplicate:", item["text"])
        for item in corpus_hits[:20]:
            print("  lm exact:", item["text"])
        raise SystemExit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for number, row in enumerate(built, start=1):
            handle.write(
                json.dumps(
                    {
                        **row,
                        "id": f"{args.prefix}-{number:04d}",
                        "reviewed": False,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    report["out"] = {"path": str(args.out), "rows": len(built)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {args.out}: {len(built)} rows {dict(sorted(by_category.items()))} "
        f"words {report['words']}"
    )


if __name__ == "__main__":
    main()
