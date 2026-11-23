"""E-022/E-026 keyterm-bias probe on a release.

Runs the product release on a mixed manifest with no context, the clip's own
English terms (oracle, optionally a seeded fraction of them), oracle +
distractors, or distractors only, and writes hypotheses for scoring.

    uv run --frozen --project ../moth-stt --extra training -- \\
      python tools/keyterm_probe.py \\
      --release runs/E-023/release-0.9.7 \\
      --manifest ../moth-cs-dataset/manifests/R-S1-cal.jsonl \\
      --audio-root ../moth-cs-dataset --context oracle \\
      --bonus 1.0 --prefix-bonus 0.25 --out-dir runs/E-022/oracle

Oracle lists contain the answers and are report-only; never a gate readout.
Distractor-only lists (E-026 condition A) contain no word of the reference.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MOTH = ROOT.parent / "moth-stt"

TOKENIZER = ROOT / "data/tokenizer/moth_bpe.model"
TERMS = ROOT.parent / "moth-stt" / "data" / "en_terms.txt"

CONTEXTS = ("none", "oracle", "oracle-distractors", "distractors")


def _runtime() -> SimpleNamespace:
    """Import the archived Moth runtime with the MOTH-50 budget and LM patches."""
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
    original_load = BinaryBPELM.load.__func__

    def _load_lm(cls, path):
        data = Path(path).read_bytes()
        if data[:8] == b"MOTHLMB2":
            return moth_lm.MothLMB2(data)
        return original_load(cls, path)

    BinaryBPELM.load = classmethod(_load_lm)
    return SimpleNamespace(
        ContextualBias=ContextualBias,
        ReleaseMoth=ReleaseMoth,
        load_audio_16k=load_audio_16k,
        normalize=normalize,
        load_tokenizer=load_tokenizer,
    )


def select_terms(
    oracle: list[str],
    reference_words: list[str],
    candidates: list[str],
    *,
    context: str,
    recall: float,
    distractors: int,
    seed: int,
    clip_id: str,
) -> tuple[list[str], list[str]]:
    """Return the kept oracle terms and the distractors for one clip.

    ``oracle`` holds the clip's normalized English terms. With ``recall < 1``
    each unique oracle term is kept with that probability (its own seeded
    stream, so the distractor draw is unchanged). Distractors never include an
    oracle term; in the ``distractors`` context they also never include any
    word of the reference.
    """
    if context not in CONTEXTS:
        raise ValueError(f"unknown context {context!r}")
    kept: list[str] = []
    if context in ("oracle", "oracle-distractors"):
        if recall >= 1.0:
            kept = list(oracle)
        else:
            rng = random.Random(f"{seed}:recall:{clip_id}")
            keep = {term for term in dict.fromkeys(oracle) if rng.random() < recall}
            kept = [term for term in oracle if term in keep]
    chosen: list[str] = []
    if context in ("oracle-distractors", "distractors"):
        exclude = set(oracle)
        if context == "distractors":
            exclude |= set(reference_words)
        pool = [term for term in candidates if term not in exclude]
        if len(pool) < distractors:
            raise ValueError(f"only {len(pool)} distractors available for {clip_id}")
        chosen = random.Random(f"{seed}:{clip_id}").sample(pool, distractors)
    return kept, chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--context", choices=CONTEXTS, default="none")
    parser.add_argument("--distractors", type=int, default=500)
    parser.add_argument(
        "--recall",
        type=float,
        default=1.0,
        help="probability of keeping each oracle term (E-026 partial-recall lists)",
    )
    parser.add_argument("--bonus", type=float, default=1.0)
    parser.add_argument("--prefix-bonus", type=float, default=0.25)
    parser.add_argument(
        "--segment", action="store_true", help="use release segmentation (default off)"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not 0.0 <= args.recall <= 1.0:
        parser.error("--recall must be in [0, 1]")

    rt = _runtime()
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = rt.load_tokenizer(TOKENIZER)

    def encode(term: str) -> tuple[int, ...]:
        return tuple(i + 1 for i in tokenizer.encode(term, out_type=int))

    runtime = rt.ReleaseMoth(args.release, budget_bytes=50_000_000)
    pieces = runtime.model.pieces
    candidates: list[str] = []
    if args.context in ("oracle-distractors", "distractors"):
        candidates = [
            rt.normalize(line)
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
    oracle_terms = 0
    term_count = 0
    distractors_used = 0
    hyps = []
    for index, row in enumerate(rows, 1):
        try:
            words = row["norm"].split()
            oracle = []
            for start, end in row.get("en_spans") or []:
                oracle.extend(words[start:end])
            oracle = [rt.normalize(term) for term in oracle]
            oracle_terms += len(dict.fromkeys(oracle))
            kept, chosen = select_terms(
                oracle,
                words,
                candidates,
                context=args.context,
                recall=args.recall,
                distractors=args.distractors,
                seed=args.seed,
                clip_id=row["id"],
            )
            terms = kept + chosen
            contextual = None
            if args.context != "none" and terms:
                accepted = []
                for term in dict.fromkeys(terms):
                    try:
                        rt.ContextualBias.from_terms(
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
                kept_set, chosen_set = set(kept), set(chosen)
                term_count += sum(term in kept_set for term in accepted)
                distractors_used += sum(term in chosen_set for term in accepted)
                if accepted:
                    contextual = rt.ContextualBias.from_terms(
                        accepted,
                        pieces,
                        encode,
                        bonus=args.bonus,
                        prefix_bonus=args.prefix_bonus,
                    )
            pcm = rt.load_audio_16k(args.audio_root / row["audio"])
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
    uses_distractors = args.context in ("oracle-distractors", "distractors")
    meta = {
        "tool": "tools/keyterm_probe.py",
        "release": str(args.release),
        "manifest": str(args.manifest),
        "audio_root": str(args.audio_root),
        "context": args.context,
        "recall": args.recall,
        "bonus": args.bonus,
        "prefix_bonus": args.prefix_bonus,
        "segment": args.segment,
        "seed": args.seed,
        "distractor_pool": len(candidates) if uses_distractors else 0,
        "distractors_requested": args.distractors if uses_distractors else 0,
        "distractors_used": distractors_used,
        "clips": len(hyps),
        "oracle_terms": oracle_terms,
        "terms_used": term_count,
        "terms_skipped": skipped_terms,
        "errors": errors,
        "audio_seconds": round(audio_seconds, 2),
        "wall_seconds": round(wall, 2),
        "rtfx": round(audio_seconds / wall, 3) if wall else None,
        "output": str(out),
        "note": (
            "distractor-only context: no reference word in the list"
            if args.context == "distractors"
            else "oracle context contains the answers; report-only, never a gate"
        ),
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
