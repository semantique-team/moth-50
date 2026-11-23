"""E-022 keyterm-bias probe on a release.

Runs the product release on R-S1-cal with no context, the clip's own English
terms (oracle), or oracle + distractors, and writes hypotheses for scoring.

    uv run --frozen --project ../moth-stt --extra training -- \\
      python tools/keyterm_probe.py \\
      --release runs/E-023/release-0.9.7 \\
      --manifest ../moth-cs-dataset/manifests/R-S1-cal.jsonl \\
      --audio-root ../moth-cs-dataset --context oracle \\
      --bonus 1.0 --prefix-bonus 0.25 --out-dir runs/E-022/oracle

The oracle contains the answers and is report-only; never a gate readout.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOTH = ROOT.parent / "moth-stt"
sys.path.insert(0, str(MOTH))

import moth_lm
from stt import onnx_runner
from stt.data.build_mel import load_audio_16k
from stt.hotwords import ContextualBias
from stt.lm import BinaryBPELM
from stt.onnx_runner import ReleaseMoth
from stt.text import normalize
from stt.tokenizer import load_tokenizer

sys.path.insert(0, str(ROOT / "vendor" / "moth-stt"))
import moth_release

# Moth's archived release.py caps the budget at 25 MB; the vendored copy is the
# MOTH-50 packaging rule (D5) and accepts the 50 MB ceiling.
onnx_runner.validate_release = moth_release.validate_release

# The archived BinaryBPELM only reads MOTHLMB1; the product LM is MOTHLMB2.
# MothLMB2 implements the exact same score() semantics (D20).
_original_binary_load = BinaryBPELM.load.__func__


def _load_lm(cls, path):
    data = Path(path).read_bytes()
    if data[:8] == b"MOTHLMB2":
        return moth_lm.MothLMB2(data)
    return _original_binary_load(cls, path)


BinaryBPELM.load = classmethod(_load_lm)

TOKENIZER = ROOT / "data/tokenizer/moth_bpe.model"
TERMS = ROOT.parent / "moth-stt" / "data" / "en_terms.txt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--context", choices=["none", "oracle", "oracle-distractors"], default="none"
    )
    parser.add_argument("--distractors", type=int, default=500)
    parser.add_argument("--bonus", type=float, default=1.0)
    parser.add_argument("--prefix-bonus", type=float, default=0.25)
    parser.add_argument(
        "--segment", action="store_true", help="use release segmentation (default off)"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(TOKENIZER)

    def encode(term: str) -> tuple[int, ...]:
        return tuple(i + 1 for i in tokenizer.encode(term, out_type=int))

    runtime = ReleaseMoth(args.release, budget_bytes=50_000_000)
    pieces = runtime.model.pieces
    candidates: list[str] = []
    if args.context == "oracle-distractors":
        candidates = [
            normalize(line)
            for line in TERMS.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        candidates = [term for term in dict.fromkeys(candidates) if term and " " not in term]
        if len(candidates) < args.distractors + 8:
            raise SystemExit(f"only {len(candidates)} distractors available")

    started = time.time()
    audio_seconds = 0.0
    errors = []
    skipped_terms = 0
    term_count = 0
    distractors_used = 0
    hyps = []
    for index, row in enumerate(rows, 1):
        try:
            words = row["norm"].split()
            terms = []
            for start, end in row.get("en_spans") or []:
                terms.extend(words[start:end])
            terms = [normalize(term) for term in terms]
            clip_distractors: list[str] = []
            if args.context == "oracle-distractors":
                own = set(terms)
                pool = [term for term in candidates if term not in own]
                clip_distractors = random.Random(f"{args.seed}:{row['id']}").sample(
                    pool, args.distractors
                )
                terms.extend(clip_distractors)
            contextual = None
            if args.context != "none" and terms:
                accepted = []
                for term in dict.fromkeys(terms):
                    try:
                        ContextualBias.from_terms(
                            [term],
                            pieces,
                            encode,
                            bonus=args.bonus,
                            prefix_bonus=args.prefix_bonus,
                        )
                    except ValueError:
                        skipped_terms += 1
                        continue
                    accepted.append(term)
                term_count += len(accepted) - len(clip_distractors)
                distractors_used += len(clip_distractors)
                if accepted:
                    contextual = ContextualBias.from_terms(
                        accepted,
                        pieces,
                        encode,
                        bonus=args.bonus,
                        prefix_bonus=args.prefix_bonus,
                    )
            pcm = load_audio_16k(args.audio_root / row["audio"])
            audio_seconds += len(pcm) / 16000
            clip_started = time.perf_counter()
            text, _ = runtime.transcribe(
                pcm,
                sample_rate=16000,
                segment=args.segment,
                contextual=contextual,
            )
            latency = time.perf_counter() - clip_started
            hyps.append({"id": row["id"], "hyp": text, "latency_s": round(latency, 4)})
        except Exception as exc:  # noqa: BLE001 - per-clip error
            errors.append({"id": row.get("id"), "error": f"{type(exc).__name__}: {exc}"})
        if index % 100 == 0:
            print(f"{index}/{len(rows)} clips", flush=True)

    wall = time.time() - started
    out = args.out_dir / "hyp.jsonl"
    out.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in hyps), encoding="utf-8"
    )
    meta = {
        "tool": "tools/keyterm_probe.py",
        "release": str(args.release),
        "manifest": str(args.manifest),
        "audio_root": str(args.audio_root),
        "context": args.context,
        "bonus": args.bonus,
        "prefix_bonus": args.prefix_bonus,
        "segment": args.segment,
        "distractors_requested": args.distractors if args.context == "oracle-distractors" else 0,
        "distractors_used": distractors_used,
        "clips": len(hyps),
        "terms_used": term_count,
        "terms_skipped": skipped_terms,
        "errors": errors,
        "audio_seconds": round(audio_seconds, 2),
        "wall_seconds": round(wall, 2),
        "rtfx": round(audio_seconds / wall, 3) if wall else None,
        "output": str(out),
        "note": "oracle context contains the answers; report-only, never a gate",
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: meta[k]
                for k in ("context", "clips", "terms_used", "terms_skipped", "errors", "rtfx")
            },
            indent=2,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
