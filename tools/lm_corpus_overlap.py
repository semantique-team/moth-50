"""Measure how much of an evaluation script set the E-031 LM corpus has seen.

The unpruned E-034 LM keeps singleton bigrams and trigrams that A3's pruning
(count >= 2) dropped. If S1-dev's cues share sentences, phrases or terms with
the corpus the LM was built from, its gain over A3 overstates what a new
speaker or new text will see (review 2026-11-21, P2). This tool reports, after
Moth's ``normalize``:

- exact normalized sentence matches;
- the share of word 4-grams present in the corpus (unique and token-weighted);
- for every English term in the cues (``en_spans``), its corpus count as a word
  sequence, bucketed 0 / 1 / >= 2. A count of 1 on a multi-word term is a term
  that only the unpruned LM can have.

Word-sequence counts are a proxy for the LM's BPE n-gram counts; they are used
because they are explainable from the scripts alone, without the tokenizer.

    python tools/lm_corpus_overlap.py \\
      --corpus ../moth-stt/runs/E-031/frozen/lm-corpus.jsonl \\
      --set s1-dev=../moth-cs-dataset/text-2026-11-dev/R-S1.jsonl \\
      --set t=../moth-cs-dataset/text/T.jsonl \\
      --out runs/E-004/lm-corpus-overlap.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vendor" / "moth-stt"))

from moth_text import normalize


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def term_tuples(row: dict) -> list[tuple[str, ...]]:
    """The row's English terms as normalized word tuples (``en_spans`` are token spans)."""
    tokens = row["text"].split()
    terms = []
    for start, end in row.get("en_spans") or []:
        term = normalize(" ".join(tokens[start:end]))
        if term:
            terms.append(tuple(term.split()))
    return terms


def analyse_set(
    rows: list[dict],
    exact: set[str],
    four_grams: set[tuple[str, ...]],
    term_counts: Counter[tuple[str, ...]],
) -> dict:
    sentences = [normalize(row["text"]) for row in rows]
    exact_hits = [row["id"] for row, text in zip(rows, sentences) if text in exact]

    unique_4grams: set[tuple[str, ...]] = set()
    total_4grams = 0
    hit_4grams = 0
    shared_4grams: Counter[tuple[str, ...]] = Counter()
    for text in sentences:
        words = text.split()
        for i in range(len(words) - 3):
            gram = tuple(words[i : i + 4])
            unique_4grams.add(gram)
            total_4grams += 1
            if gram in four_grams:
                hit_4grams += 1
                shared_4grams[gram] += 1

    terms: list[tuple[str, ...]] = []
    for row in rows:
        terms.extend(term_tuples(row))

    def bucket(terms_subset: list[tuple[str, ...]]) -> dict:
        counts = Counter(term_counts[t] for t in terms_subset)
        n = len(terms_subset)
        return {
            "occurrences": n,
            "zero": counts[0],
            "one": counts[1],
            "two_plus": sum(v for k, v in counts.items() if k >= 2),
            "one_share": round(counts[1] / n, 4) if n else None,
            "zero_share": round(counts[0] / n, 4) if n else None,
        }

    unique_terms = sorted(set(terms))
    multiword = [t for t in terms if len(t) >= 2]
    singletons = sorted(t for t in unique_terms if term_counts[t] == 1)
    absent = sorted(t for t in unique_terms if term_counts[t] == 0)

    return {
        "rows": len(rows),
        "words": sum(len(text.split()) for text in sentences),
        "exact_matches": len(exact_hits),
        "exact_match_ids": exact_hits[:20],
        "four_grams": {
            "unique": len(unique_4grams),
            "unique_present": sum(1 for gram in unique_4grams if gram in four_grams),
            "unique_share": (
                round(
                    sum(1 for gram in unique_4grams if gram in four_grams) / len(unique_4grams), 4
                )
                if unique_4grams
                else None
            ),
            "occurrences": total_4grams,
            "occurrences_present": hit_4grams,
            "occurrence_share": round(hit_4grams / total_4grams, 4) if total_4grams else None,
            "top_shared": [
                {"gram": " ".join(gram), "count": count}
                for gram, count in shared_4grams.most_common(10)
            ],
        },
        "terms": {
            "unique": len(unique_terms),
            "unique_multiword": len([t for t in unique_terms if len(t) >= 2]),
            "all": bucket(terms),
            "multiword": bucket(multiword),
            "one_examples": [" ".join(t) for t in singletons[:15]],
            "zero_examples": [" ".join(t) for t in absent[:15]],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", type=Path, required=True, help="E-031 frozen lm-corpus.jsonl")
    parser.add_argument(
        "--set",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="script JSONL with text and en_spans; repeatable",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    corpus_rows = load_rows(args.corpus)
    # ``mix`` is the code-switch part of the corpus; normalize() only takes one
    # spelling language, and Italian is the corpus majority.
    corpus_sentences = [
        normalize(row["text"], "it" if row.get("lang") == "mix" else row.get("lang", "it"))
        for row in corpus_rows
    ]
    exact = set(corpus_sentences)
    four_grams: set[tuple[str, ...]] = set()
    token_lists = []
    for text in corpus_sentences:
        words = text.split()
        token_lists.append(words)
        four_grams.update(tuple(words[i : i + 4]) for i in range(len(words) - 3))

    sets = {}
    for spec in args.set:
        label, _, raw_path = spec.partition("=")
        sets[label] = load_rows(Path(raw_path))

    terms = {t for rows in sets.values() for row in rows for t in term_tuples(row)}
    term_counts: Counter[tuple[str, ...]] = Counter()
    for words in token_lists:
        for n in range(1, max((len(t) for t in terms), default=1) + 1):
            for i in range(len(words) - n + 1):
                gram = tuple(words[i : i + n])
                if gram in terms:
                    term_counts[gram] += 1

    report = {
        "tool": "tools/lm_corpus_overlap.py",
        "corpus": {
            "path": str(args.corpus),
            "sha256": sha256(args.corpus),
            "sentences": len(corpus_rows),
            "words": sum(len(text.split()) for text in corpus_sentences),
        },
        "normalizer": "vendor/moth-stt/moth_text.normalize",
        "lang_hints": {"mix": "it"},
        "sets": {
            label: analyse_set(rows, exact, four_grams, term_counts) for label, rows in sets.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for label, result in report["sets"].items():
        grams = result["four_grams"]
        terms_report = result["terms"]
        print(
            f"{label}: {result['rows']} rows, exact {result['exact_matches']}, "
            f"4-gram unique {grams['unique_share']} / occurrence {grams['occurrence_share']}, "
            f"multiword terms at count 1: {terms_report['multiword']['one_share']}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
