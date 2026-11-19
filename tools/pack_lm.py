"""Pack a MOTHLMB1 LM as lossless MOTHLMB2 and verify the round trip exactly.

    python tools/pack_lm.py --lm data/raw/moth-0.8.1/lm_bpe3.bin \
        --out runs/E-012/lm_a3_mothlmb2.bin --bits 10 \
        --report runs/E-012/lm-pack.json

The tool refuses a lossy codebook: every distinct int16 log-prob must fit in
``2**bits``. It then decodes the packed file and checks every array against the
source (tokens, log-probs, row offsets, trigram keys, unigram, backoff and log
scale), so a successful run is a proof that scores cannot change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from moth_lm import packed_size, read_mothlmb1, read_mothlmb2, write_mothlmb2

ARRAYS = (
    "unigram",
    "bigram_offsets",
    "bigram_tokens",
    "bigram_logq",
    "trigram_keys",
    "trigram_offsets",
    "trigram_tokens",
    "trigram_logq",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pack_lm.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lm", type=Path, required=True, help="input MOTHLMB1 file")
    parser.add_argument("--out", type=Path, required=True, help="output MOTHLMB2 file")
    parser.add_argument(
        "--bits", type=int, default=10, help="per-order codebook width (default 10)"
    )
    parser.add_argument("--report", type=Path, default=None, help="JSON report path")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if args.report is not None and args.report.exists():
        raise SystemExit(f"refusing to overwrite {args.report}")

    source = args.lm.read_bytes()
    lm = read_mothlmb1(source)
    packed = write_mothlmb2(lm, bits=args.bits)
    decoded = read_mothlmb2(packed)

    for key in ARRAYS:
        if not np.array_equal(decoded[key].astype(np.int64), lm[key].astype(np.int64)):
            raise SystemExit(f"round trip failed for {key}")
    if decoded["backoff"] != lm["backoff"] or decoded["log_scale"] != lm["log_scale"]:
        raise SystemExit("round trip failed for the header")

    counts = {
        "bigrams": len(lm["bigram_tokens"]),
        "trigrams": len(lm["trigram_tokens"]),
        "contexts": len(lm["trigram_keys"]),
    }
    expected = packed_size(
        counts["bigrams"], counts["trigrams"], counts["contexts"], bits=args.bits
    )
    if len(packed) != expected:
        raise SystemExit(f"size formula {expected} != written {len(packed)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(packed)
    report = {
        "tool": "tools/pack_lm.py",
        "source": {
            "path": str(args.lm),
            "bytes": len(source),
            "sha256": sha256(source),
            "distinct_bigram_values": int(np.unique(lm["bigram_logq"]).size),
            "distinct_trigram_values": int(np.unique(lm["trigram_logq"]).size),
            **counts,
        },
        "packed": {
            "path": str(args.out),
            "bytes": len(packed),
            "sha256": sha256(packed),
            "bits": args.bits,
            "lossless": True,
            "saved_bytes": len(source) - len(packed),
            "ratio": round(len(packed) / len(source), 5),
        },
        "verification": {
            "arrays_equal": list(ARRAYS),
            "header_equal": ["backoff", "log_scale"],
            "size_formula_matches": True,
        },
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["packed"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
