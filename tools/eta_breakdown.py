"""Break down the English-term (ETA) misses of a frozen readout.

``moth_eval`` lists every missed English reference word as ``{"id", "term", "got"}``
(``got`` is the aligned hypothesis word, or ``None`` for a deletion). This tool
classifies those misses, flags the term shapes that the 2026-11-19 S1-dev review
found hardest, and optionally compares two readouts of the same set:

    python tools/eta_breakdown.py runs/E-005/s1-dev-a3-int8.json \
      --compare runs/E-005/s1-dev-0.8.1.json --out runs/E-005/eta-breakdown.json

Classes (exactly one per miss, checked in this order):

- ``deleted``: no hypothesis word aligned (``got`` is ``None``);
- ``apostrophe``: identical once apostrophes are removed (``lopex`` for ``l'opex``);
- ``glued``: the term is a strict substring of the hypothesis word, i.e. it was
  fused with a neighbour (``storagebucket`` for ``bucket``);
- ``truncated``: the hypothesis word is a strict prefix of the term (``pat``);
- ``substituted``: anything else.

Flags (independent, any number per miss):

- ``elided``: the term carries an apostrophe (``l'sdk``, ``let's``);
- ``no_vowel``: after an optional elided prefix the term has no vowel
  (``mcv``, ``l'hr``) — a lower bound on spelled acronyms, since ``cig`` or
  ``ios`` have vowels;
- ``function_word``: the term is in a fixed list of English closed-class words.

It reads only the readout JSON (and ``--manifest`` for per-category counts), so
it runs anywhere the readouts are, without audio or a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

KINDS = ("deleted", "apostrophe", "glued", "truncated", "substituted")
FLAGS = ("elided", "no_vowel", "function_word")

_FUNCTION_WORD_GROUPS = (
    "a an the",
    "i you he she it we they me him her us them my your his its our their",
    "this that these those",
    "of in on at to from by for with about into onto over under up down off out",
    "and or but so than then if as",
    "not no too very",
    "is are was were be been being am do does did have has had",
    "will would can could shall should may might must",
)
#: English closed-class words: articles, pronouns, prepositions and particles,
#: conjunctions, auxiliaries and modals. Content words are deliberately absent.
FUNCTION_WORDS = frozenset(word for group in _FUNCTION_WORD_GROUPS for word in group.split())

_NO_VOWEL = re.compile(r"(?:[a-z]+')?[^aeiouy'\s]{2,}")


def classify(term: str, got: str | None) -> str:
    if not got:
        return "deleted"
    if term.replace("'", "") == got.replace("'", ""):
        return "apostrophe"
    if term.replace("'", "") in got.replace("'", ""):
        return "glued"
    if len(got) < len(term) and term.startswith(got):
        return "truncated"
    return "substituted"


def flags(term: str) -> list[str]:
    found = []
    if "'" in term:
        found.append("elided")
    if _NO_VOWEL.fullmatch(term) and any(char.isalpha() for char in term):
        found.append("no_vowel")
    if term in FUNCTION_WORDS:
        found.append("function_word")
    return found


def summarize(errors: list[dict], categories: dict[str, str] | None = None) -> dict:
    """Counts per class and flag, top missed terms and examples."""
    kinds: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    by_category: dict[str, Counter[str]] = {}
    examples: dict[str, list[list]] = {kind: [] for kind in KINDS}
    for error in errors:
        kind = classify(error["term"], error["got"])
        kinds[kind] += 1
        for flag in flags(error["term"]):
            flag_counts[flag] += 1
        if len(examples[kind]) < 15:
            examples[kind].append([error["term"], error["got"]])
        if categories is not None:
            category = categories.get(str(error["id"]), "unknown")
            by_category.setdefault(category, Counter())[kind] += 1
    return {
        "misses": len(errors),
        "kinds": {kind: kinds[kind] for kind in KINDS},
        "flags": {flag: flag_counts[flag] for flag in FLAGS},
        "top_terms": Counter(error["term"] for error in errors).most_common(20),
        "examples": examples,
        "by_category": (
            {name: dict(counts) for name, counts in sorted(by_category.items())}
            if categories is not None
            else None
        ),
    }


def compare(errors: list[dict], other: list[dict]) -> dict:
    """Misses of ``errors`` that ``other`` does not have, and the reverse, by (id, term)."""
    ours = Counter((str(error["id"]), error["term"]) for error in errors)
    theirs = Counter((str(error["id"]), error["term"]) for error in other)
    only_ours = ours - theirs
    only_theirs = theirs - ours
    got = {(str(error["id"]), error["term"]): error["got"] for error in errors}
    return {
        "misses": len(errors),
        "other_misses": len(other),
        "only_here": sum(only_ours.values()),
        "only_other": sum(only_theirs.values()),
        "only_here_examples": [[term, got[(uid, term)]] for uid, term in list(only_ours)[:30]],
    }


def reference_counts(rows: list[dict]) -> dict:
    """English reference words per flag and per category, from the set manifest.

    These are the ETA denominators, so ``1 - misses / total`` is the accuracy of a
    flag or category. Words are taken from ``norm`` at the ``en_spans`` indices,
    exactly as ``moth_eval`` scores them.
    """
    per_flag: Counter[str] = Counter()
    per_category: Counter[str] = Counter()
    total = 0
    for row in rows:
        words = (row.get("norm") or "").split()
        for start, end in row.get("en_spans") or []:
            for term in words[start:end]:
                total += 1
                per_category[str(row.get("category") or row.get("lang"))] += 1
                for flag in flags(term):
                    per_flag[flag] += 1
    return {
        "english_words": total,
        "flags": {flag: per_flag[flag] for flag in FLAGS},
        "categories": dict(sorted(per_category.items())),
    }


def category_table(readout: dict) -> dict:
    table = readout["metrics"].get("category") or {}
    return {
        name: {key: row.get(key) for key in ("n", "ref_words", "sub", "del", "ins", "wer", "eta")}
        for name, row in table.items()
    }


def _sha256(path: Path) -> str:
    """sha256 of the file with CRLF normalized to LF, i.e. of the content git stores."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _load(path: Path) -> dict:
    readout = json.loads(path.read_text(encoding="utf-8"))
    if "metrics" not in readout or "eta_errors" not in readout["metrics"]:
        raise SystemExit(f"{path} is not a --mode mixed readout (no metrics.eta_errors)")
    return readout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eta_breakdown.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("readout", type=Path, help="frozen_readout.py --mode mixed JSON")
    parser.add_argument("--compare", type=Path, help="a second readout of the same set")
    parser.add_argument("--manifest", type=Path, help="set manifest, for per-category misses")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.out and args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    readout = _load(args.readout)
    categories = None
    references = None
    if args.manifest:
        rows = [
            json.loads(line)
            for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        categories = {str(row["id"]): str(row.get("category") or row.get("lang")) for row in rows}
        references = reference_counts(rows)
    output = {
        "tool": "tools/eta_breakdown.py",
        "readout": {
            "path": args.readout.as_posix(),
            "sha256": _sha256(args.readout),
            "eta": readout["metrics"]["eta"],
            "wer": readout["metrics"]["wer"],
        },
        "manifest": (
            {"path": args.manifest.as_posix(), "sha256": _sha256(args.manifest)}
            if args.manifest
            else None
        ),
        "reference_counts": references,
        "summary": summarize(readout["metrics"]["eta_errors"], categories),
        "categories": category_table(readout),
        "compare": None,
    }
    if args.compare:
        other = _load(args.compare)
        output["compare"] = {
            "readout": {
                "path": args.compare.as_posix(),
                "sha256": _sha256(args.compare),
                "eta": other["metrics"]["eta"],
                "wer": other["metrics"]["wer"],
            },
            "summary": summarize(other["metrics"]["eta_errors"], categories),
            "categories": category_table(other),
            "diff": compare(readout["metrics"]["eta_errors"], other["metrics"]["eta_errors"]),
        }
    text = json.dumps(output, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    summary = output["summary"]
    print(f"misses {summary['misses']}: {summary['kinds']} flags {summary['flags']}")
    if output["compare"]:
        diff = output["compare"]["diff"]
        print(f"vs {args.compare}: only here {diff['only_here']}, only there {diff['only_other']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
