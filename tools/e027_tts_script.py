"""E-027 TTS script: the E-025 filtered sentences prepared for synthesis.

Re-filters ``runs/E-025/filtered.jsonl`` for text-to-speech and tags every
surface token as English or Italian for the mixed-G2P rendering:

- drop sentences with digits (the normalizer and the voices would read them
  differently, so the transcript could not be trusted);
- keep 4-30 normalized words;
- drop every sentence that shares a normalized word 4-gram with S1-dev, T, A
  **or R-S1-cal** (E-025 checked the first three only), and exact duplicates.

A word is English if it is a listed term (Moth's ``en_terms.txt``) or an
English function word, or if it occurs in the English training text (the
gold-train English spans, LibriSpeech-100) and is not a frequent Italian
training word. ``control_en`` rows are English throughout and ``control_it``
rows Italian throughout. An elided Italian prefix (``l'``, ``dell'``...) is
split off when the rest of the token is English (``l'SDK`` -> ``l'`` + ``SDK``).

    uv run --frozen --project ../moth-stt --extra training -- \\
      python tools/e027_tts_script.py --out runs/E-027/tts-script.jsonl \\
      --report runs/E-027/tts-script-report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOTH = ROOT.parent / "moth-stt"
DATASET = ROOT.parent / "moth-cs-dataset"

SOURCE = ROOT / "runs" / "E-025" / "filtered.jsonl"
EVAL_MANIFESTS = (
    DATASET / "manifests" / "R-S1-dev.jsonl",
    DATASET / "manifests" / "test.jsonl",
    DATASET / "manifests" / "anchor.jsonl",
    DATASET / "manifests" / "R-S1-cal.jsonl",
)
EN_TERMS = MOTH / "data" / "en_terms.txt"
GOLD_TRAIN = DATASET / "manifests" / "train.jsonl"
EN_TEXT = (MOTH / "data" / "manifests" / "ls_en_train.jsonl",)
IT_TEXT = (
    MOTH / "data" / "manifests" / "mls_it_train.jsonl",
    MOTH / "data" / "manifests" / "cv_it_train_clean50.jsonl",
)

#: An Italian training word at least this frequent vetoes the English tag,
#: unless the word is a listed term (``file``, ``computer``...).
IT_MIN_COUNT = 3

#: English function words that are not also Italian words (``a``, ``i``, ``in``,
#: ``do``, ``so``, ``me`` and ``more`` are left out on purpose).
# fmt: off
ENGLISH_FUNCTION = frozenset({
    "the", "an", "and", "or", "of", "to", "on", "for", "with", "is", "are", "was",
    "were", "this", "that", "it", "we", "you", "they", "have", "has", "not", "my",
    "our", "your", "at", "by", "from", "be", "been", "will", "would", "can", "could",
    "should", "does", "did", "what", "when", "where", "which", "who", "how", "there",
    "here", "about", "just", "all", "some", "any", "most", "very", "too", "also",
    "only", "then", "than", "into", "out", "up", "down", "over", "after", "before",
    "again", "because", "if", "but", "its", "their", "his", "her", "him", "them",
    "us", "i'm", "it's", "don't", "can't", "let's",
})
# fmt: on

ELISION = re.compile(
    r"^((?:l|un|dell|all|dall|nell|sull|coll|quell|quest|d|c|s|m|t|v|n)')(.+)$",
    re.IGNORECASE,
)
EDGE_PUNCT = '.,;:!?"()[]{}«»“”‘…-–—'
DIGIT = re.compile(r"\d")


@dataclass(frozen=True)
class Lexicon:
    terms: frozenset[str]
    english: frozenset[str]
    italian: frozenset[str]

    def is_english(self, word: str) -> bool:
        if word in self.terms or word in ENGLISH_FUNCTION:
            return True
        return word in self.english and word not in self.italian


def has_digit(text: str) -> bool:
    return bool(DIGIT.search(text))


def core(token: str) -> str:
    """The token without edge punctuation, with typographic apostrophes unified."""
    return token.replace("’", "'").strip(EDGE_PUNCT)


def split_elision(token: str) -> tuple[str, str] | None:
    """``("l'", "SDK,")`` for ``"l'SDK,"``; ``None`` if the token is not elided."""
    unified = token.replace("’", "'")
    match = ELISION.match(unified)
    if not match:
        return None
    return match.group(1), match.group(2)


def word_is_english(surface: str, lexicon: Lexicon, normalize: Callable[[str], str]) -> bool:
    words = normalize(core(surface)).split()
    return bool(words) and all(lexicon.is_english(word) for word in words)


def segments(
    text: str,
    category: str | None,
    lexicon: Lexicon,
    normalize: Callable[[str], str],
) -> list[list[str]]:
    """Surface pieces with a language tag, in order: ``[[piece, "en"|"it"], ...]``."""
    if category == "control_en":
        return [[token, "en"] for token in text.split()]
    if category == "control_it":
        return [[token, "it"] for token in text.split()]
    pieces: list[list[str]] = []
    for token in text.split():
        if word_is_english(token, lexicon, normalize):
            pieces.append([token, "en"])
            continue
        elided = split_elision(token)
        if elided and word_is_english(elided[1], lexicon, normalize):
            pieces.extend([[elided[0], "it"], [elided[1], "en"]])
            continue
        pieces.append([token, "it"])
    return pieces


def row_lang(category: str | None) -> str:
    return {"control_en": "en", "control_it": "it"}.get(category or "", "mix")


def four_grams(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + 4]) for i in range(len(words) - 3)}


def keep_reason(
    text: str, norm: str, eval_grams: set[tuple[str, ...]], seen: set[str]
) -> str | None:
    """The drop reason for one sentence, or ``None`` to keep it."""
    if has_digit(text):
        return "digits"
    words = norm.split()
    if not 4 <= len(words) <= 30:
        return "length"
    if four_grams(words) & eval_grams:
        return "eval_4gram"
    if norm in seen:
        return "duplicate"
    return None


def script_id(source_id: str) -> str:
    """``E025-00001`` -> ``E027-00001``."""
    return "E027-" + source_id.split("-", 1)[1]


def _jsonl(path: Path) -> Iterable[dict]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lexicon(normalize: Callable[[str], str]) -> Lexicon:
    terms = {
        word
        for line in EN_TERMS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
        for word in normalize(line).split()
    }
    english: set[str] = set()
    for row in _jsonl(GOLD_TRAIN):
        words = row["norm"].split()
        for start, end in row.get("en_spans") or []:
            english.update(words[start:end])
    for path in EN_TEXT:
        for row in _jsonl(path):
            english.update(row["norm"].split())
    counts: Counter[str] = Counter()
    for path in IT_TEXT:
        for row in _jsonl(path):
            counts.update(row["norm"].split())
    italian = {word for word, count in counts.items() if count >= IT_MIN_COUNT}
    return Lexicon(frozenset(terms), frozenset(english), frozenset(italian))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT / "vendor" / "moth-stt"))
    from moth_text import normalize

    eval_grams: set[tuple[str, ...]] = set()
    for path in EVAL_MANIFESTS:
        for row in _jsonl(path):
            eval_grams |= four_grams(normalize(row.get("norm") or row["text"]).split())
    lexicon = _lexicon(normalize)

    kept = []
    reasons: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    seen: set[str] = set()
    en_pieces = 0
    for row in _jsonl(args.source):
        text = " ".join(row["text"].split())
        norm = normalize(text)
        reason = keep_reason(text, norm, eval_grams, seen)
        if reason:
            reasons[reason] += 1
            continue
        seen.add(norm)
        category = row.get("category")
        pieces = segments(text, category, lexicon, normalize)
        en_pieces += sum(tag == "en" for _, tag in pieces)
        by_category[category or "?"] += 1
        kept.append(
            {
                "id": script_id(row["id"]),
                "source_id": row["id"],
                "text": text,
                "norm": norm,
                "category": category,
                "lang": row_lang(category),
                "segments": pieces,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8"
    )
    report = {
        "tool": "tools/e027_tts_script.py",
        "source": str(args.source),
        "source_sha256": _sha256(args.source),
        "out": str(args.out),
        "out_sha256": _sha256(args.out),
        "input": len(kept) + sum(reasons.values()),
        "kept": len(kept),
        "dropped": dict(reasons),
        "by_category": dict(by_category),
        "english_pieces": en_pieces,
        "rows_with_english": sum(any(tag == "en" for _, tag in row["segments"]) for row in kept),
        "eval_sources": [str(path) for path in EVAL_MANIFESTS],
        "eval_4grams": len(eval_grams),
        "lexicon": {
            "terms": len(lexicon.terms),
            "english_words": len(lexicon.english),
            "italian_words": len(lexicon.italian),
            "italian_min_count": IT_MIN_COUNT,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
