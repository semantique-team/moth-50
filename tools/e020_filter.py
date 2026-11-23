"""Filter E-020 generated text before it enters the LM corpus.

Drops evaluation-script 4-gram overlaps, exact/near duplicates, wrong-length
lines and category mismatches. The eval scripts are S1-dev, T and A; the report
records every reason. No generated text is used unfiltered.

    uv run --frozen --project ../moth-stt --extra training -- \\
      python tools/e020_filter.py --generated runs/E-025/generated.jsonl \\
      --out runs/E-025/filtered.jsonl --report runs/E-025/filter-report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOTH = ROOT.parent / "moth-stt"
DATASET = ROOT.parent / "moth-cs-dataset"
sys.path.insert(0, str(MOTH))

from stt.text import normalize

ITALIAN = {
    "il",
    "lo",
    "la",
    "i",
    "gli",
    "le",
    "un",
    "una",
    "di",
    "che",
    "e",
    "è",
    "per",
    "con",
    "non",
    "del",
    "della",
    "al",
    "alla",
    "in",
    "su",
    "da",
    "si",
    "mi",
    "ti",
    "ci",
    "vi",
    "ho",
    "hai",
    "ha",
    "hanno",
    "sono",
    "questo",
    "questa",
    "come",
    "più",
    "anche",
    "dopo",
    "prima",
    "nel",
    "sul",
}
ENGLISH = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "this",
    "that",
    "it",
    "we",
    "you",
    "they",
    "i",
    "have",
    "has",
    "not",
    "my",
    "our",
    "your",
    "at",
    "by",
    "from",
}
ITALIANIZED = re.compile(r"\b\w+(?:ato|ata|ati|ate|izzato|izzata|are)\b", re.IGNORECASE)


def _grams(text: str, n: int = 4) -> set[tuple[str, ...]]:
    words = normalize(text).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _eval_grams() -> set[tuple[str, ...]]:
    sources = [
        DATASET / "manifests" / "R-S1-dev.jsonl",
        DATASET / "manifests" / "test.jsonl",
        DATASET / "manifests" / "anchor.jsonl",
    ]
    grams: set[tuple[str, ...]] = set()
    for path in sources:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("norm") or row.get("text") or ""
            grams |= _grams(text)
    return grams


def _keep(row: dict, eval_grams: set[tuple[str, ...]]) -> tuple[bool, str]:
    text = normalize(row["text"])
    words = text.split()
    if not 4 <= len(words) <= 30:
        return False, "length"
    if _grams(text) & eval_grams:
        return False, "eval_4gram"
    category = row.get("category")
    seeds = {normalize(term) for term in row.get("seed_terms") or [] if term}
    if category in {"single_word", "phrase", "brand", "acronym"}:
        if seeds and not (seeds & set(words)):
            return False, "seed_missing"
    elif category == "switch":
        has_italian = bool(set(words) & ITALIAN)
        has_english = bool(set(words) & ENGLISH) or bool(seeds & set(words))
        if not (has_italian and has_english):
            return False, "not_mixed"
    elif category == "italianized":
        if not ITALIANIZED.search(text):
            return False, "not_italianized"
    elif category == "control_it":
        if set(words) & ENGLISH:
            return False, "english_in_control_it"
    elif category == "control_en" and (set(words) & ITALIAN or not (set(words) & ENGLISH)):
        return False, "not_english"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    eval_grams = _eval_grams()
    rows = [
        json.loads(line)
        for line in args.generated.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kept = []
    reasons = Counter()
    by_category = Counter()
    seen_text = set()
    seen_grams: set[frozenset[tuple[str, ...]]] = set()
    for row in rows:
        ok, reason = _keep(row, eval_grams)
        if not ok:
            reasons[reason] += 1
            continue
        text = normalize(row["text"])
        if text in seen_text:
            reasons["exact_duplicate"] += 1
            continue
        grams = frozenset(_grams(text))
        if grams in seen_grams:
            reasons["near_duplicate"] += 1
            continue
        seen_text.add(text)
        seen_grams.add(grams)
        by_category[row.get("category", "?")] += 1
        kept.append({"text": row["text"], "category": row.get("category"), "id": row.get("id")})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8"
    )
    report = {
        "tool": "tools/e020_filter.py",
        "generated": str(args.generated),
        "out": str(args.out),
        "input": len(rows),
        "kept": len(kept),
        "dropped": len(rows) - len(kept),
        "reasons": dict(reasons),
        "by_category": dict(by_category),
        "eval_4grams": len(eval_grams),
        "eval_sources": [
            str(DATASET / "manifests" / "R-S1-dev.jsonl"),
            str(DATASET / "manifests" / "test.jsonl"),
            str(DATASET / "manifests" / "anchor.jsonl"),
        ],
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
