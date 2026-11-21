"""Build a MOTHLMB1 n-gram LM from a corpus, mirroring Moth E-034.

Moth E-034's ``build_variants.py`` lives in the archived ``../moth-stt``
checkout and hard-codes the frozen corpus. P2 needs the same builder on a
cleaned corpus, and E-020 will need it again. This tool imports Moth's pinned
``build_lm.weighted_sequences`` and ``stt.lm`` code instead of copying them, so
the counts, term weighting, backoff and binary writer stay identical.

    uv run --frozen --project ../moth-stt --extra training -- \\
      python tools/build_lm_from_corpus.py --corpus /tmp/lm-corpus-nos1dev.jsonl \\
      --out runs/E-004/lm_e034c_n2 --report runs/E-004/lm-build-e034c.json

Writes ``<out>.pkl.gz`` (PyTorch readout) and ``<out>.bin`` (MOTHLMB1, the
browser format and the input ``frozen_readout.py --lm`` expects). Pack the
``.bin`` with ``tools/pack_lm.py`` for the release package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MOTH = REPO.parent / "moth-stt"
sys.path.insert(0, str(MOTH))
sys.path.insert(0, str(MOTH / "runs" / "E-031"))

from build_lm import TERM_RE, weighted_sequences
from stt.lm import BPETrigramLM
from stt.tokenizer import load_tokenizer

TOKENIZER = REPO / "data/tokenizer/moth_bpe.model"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output prefix, no extension")
    parser.add_argument("--name", default=None, help="variant name for the report")
    parser.add_argument("--min-bigram", type=int, default=1)
    parser.add_argument("--min-trigram", type=int, default=1)
    parser.add_argument("--term-weight", type=int, default=1)
    parser.add_argument("--backoff", type=float, default=0.4)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    rows = [
        json.loads(line)
        for line in args.corpus.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    processor = load_tokenizer(TOKENIZER)
    tagged = [(row["text"], processor.encode(row["text"], out_type=int)) for row in rows]
    term_rows = [text for text, _ in tagged if TERM_RE.search(text)]
    print(
        f"corpus {len(rows)} sentences, {sum(len(t) for _, t in tagged)} tokens, "
        f"term sentences {len(term_rows)}",
        flush=True,
    )

    lm = BPETrigramLM.from_sequences(
        weighted_sequences(tagged, args.term_weight),
        min_bigram_count=args.min_bigram,
        min_trigram_count=args.min_trigram,
        backoff=args.backoff,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pickle_path = args.out.with_suffix(".pkl.gz")
    binary_path = args.out.with_suffix(".bin")
    lm.save(pickle_path)
    lm.save_browser(binary_path)

    report = {
        "tool": "tools/build_lm.py",
        "name": args.name or args.out.name,
        "corpus": {"path": str(args.corpus), "sha256": sha256(args.corpus), "rows": len(rows)},
        "tokenizer": {"path": str(TOKENIZER), "sha256": sha256(TOKENIZER)},
        "builder": {
            "module": "runs/E-031/build_lm.py",
            "min_bigram": args.min_bigram,
            "min_trigram": args.min_trigram,
            "term_weight": args.term_weight,
            "backoff": args.backoff,
            "term_sentences": len(term_rows),
        },
        "pickle": {
            "path": str(pickle_path),
            "bytes": pickle_path.stat().st_size,
            "sha256": sha256(pickle_path),
        },
        "binary": {
            "path": str(binary_path),
            "bytes": binary_path.stat().st_size,
            "sha256": sha256(binary_path),
            "bigram_entries": sum(len(r) for r in lm._bigram_counts.values()),
            "trigram_entries": sum(len(r) for r in lm._trigram_counts.values()),
        },
        "seconds": round(time.time() - started, 1),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"{report['name']}: binary {report['binary']['bytes']:,} B, "
        f"{report['binary']['bigram_entries']:,} bigrams / "
        f"{report['binary']['trigram_entries']:,} trigrams, {report['seconds']}s"
    )


if __name__ == "__main__":
    main()
