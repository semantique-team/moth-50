"""Remove evaluation-script sentences from an LM corpus.

The E-031 corpus included ``pool.jsonl`` (the sentence pool the S1-dev scripts
were drawn from), so 900 of 960 S1-dev cues are exact members of the corpus the
E-034 LM was built on. This tool drops every corpus sentence whose normalized
text equals a normalized script sentence, and writes a derivation report. It is
the P2 remediation in ``docs/review-2026-11-21.md``.

    python tools/clean_lm_corpus.py \\
      --corpus ../moth-stt/runs/E-031/frozen/lm-corpus.jsonl \\
      --scripts ../moth-cs-dataset/text-2026-11-dev/R-S1.jsonl \\
      --out /tmp/lm-corpus-nos1dev.jsonl \\
      --report runs/E-004/lm-corpus-clean-report.json

The output keeps the original lines byte-for-byte for the sentences it keeps,
so it is a pure subset of the frozen corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vendor" / "moth-stt"))

from moth_text import normalize


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_lines(corpus_bytes: bytes, script_texts: set[str]) -> tuple[list[str], list[str]]:
    """Split the corpus into kept original lines and dropped texts."""
    kept: list[str] = []
    dropped: list[str] = []
    for line in corpus_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        hint = "it" if row.get("lang") == "mix" else row.get("lang", "it")
        if normalize(row["text"], hint) in script_texts:
            dropped.append(row["text"])
        else:
            kept.append(line)
    return kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--scripts",
        type=Path,
        action="append",
        required=True,
        help="script JSONL whose sentences must not stay in the corpus",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    script_texts: set[str] = set()
    script_report = []
    for path in args.scripts:
        data = path.read_bytes()
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        texts = {normalize(row["text"]) for row in rows}
        script_texts |= texts
        script_report.append(
            {
                "path": str(path),
                "sha256": sha256_bytes(data),
                "rows": len(rows),
                "normalized_sentences": len(texts),
            }
        )

    corpus_bytes = args.corpus.read_bytes()
    kept, dropped = clean_lines(corpus_bytes, script_texts)

    out_bytes = ("\n".join(kept) + "\n").encode("utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(out_bytes)
    report = {
        "tool": "tools/clean_lm_corpus.py",
        "corpus": {
            "path": str(args.corpus),
            "sha256": sha256_bytes(corpus_bytes),
            "rows": sum(1 for line in corpus_bytes.decode("utf-8").splitlines() if line.strip()),
        },
        "scripts": script_report,
        "dropped_rows": len(dropped),
        "kept_rows": len(kept),
        "dropped_examples": dropped[:10],
        "out": {"path": str(args.out), "sha256": sha256_bytes(out_bytes), "rows": len(kept)},
        "normalizer": "vendor/moth-stt/moth_text.normalize",
        "lang_hints": {"mix": "it"},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"dropped {len(dropped)} of {len(dropped) + len(kept)} rows; "
        f"kept {len(kept)} -> {args.out} ({report['out']['sha256'][:12]}…)"
    )


if __name__ == "__main__":
    main()
