"""Frozen E-002 readout for a MOTH-50 checkpoint: beam 12 + A3 LM.

This is the gate decoder, not the trainer's greedy metric: it loads the E-005
checkpoint, runs the two dev sets (MLS-IT + LS-EN), decodes every clip with the
frozen settings and scores WER + a paired macro bootstrap against the retained
0.8.1 hypotheses.

    uv run --frozen --project ../moth-stt --extra training -- \
      python tools/frozen_readout.py \
      --checkpoint runs/E-005/base+6-16k/checkpoints/ckpt_best.pt \
      --config training/configs/base6-long.toml \
      --out runs/E-005/readout.json --device cpu

Fixed decoder (experiments/E-002-protocol.md): beam 12, top-k 10, lm_weight
0.55, token_bonus 2.7 and the 0.8.1 A3 LM, resolved and sha256-checked by
``tools/release_assets.py`` (``data/raw/moth-0.8.1/lm_bpe3.bin``). The dev sets
and the reference hypotheses are the same ones behind the 23.0170 % reference.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

from release_assets import resolve_lm

#: Frozen E-002 decoder settings.
BEAM_WIDTH = 12
TOP_K = 10
LM_WEIGHT = 0.55
TOKEN_BONUS = 2.7

#: E-002 protocol gate.
REFERENCE_MACRO = 0.230170
GATE_MARGIN = 0.010

REPO = Path(__file__).resolve().parents[1]
MOTH_STT = REPO.parent / "moth-stt"
DEFAULT_REFERENCE_HYP = MOTH_STT / "runs/E-037/readout/paired-1/dev/int8/dev.hyp.jsonl"


def macro_wer(per_set: dict[str, float]) -> float:
    """Equal-language macro: mean of the IT and EN set WERs."""
    if set(per_set) != {"it", "en"}:
        raise ValueError(f"macro needs exactly it/en, got {sorted(per_set)}")
    return (per_set["it"] + per_set["en"]) / 2


def gate_verdict(diff: float, ci: list[float] | None) -> dict:
    """Gate on the paired macro diff (ours - reference, negative is better)."""
    improvement = -diff
    excludes_zero = bool(ci is not None and ci[1] < 0)
    return {
        "improvement_pp": round(improvement * 100, 4),
        "ci_excludes_zero": excludes_zero,
        "meets_1pp": improvement >= GATE_MARGIN,
        "pass": improvement >= GATE_MARGIN and excludes_zero,
    }


def sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def decoder_assets(config, lm_path: Path):
    """Tokenizer pieces and the decoder LM, shared by the PyTorch and ONNX paths."""
    from stt.lm import BinaryBPELM
    from stt.tokenizer import load_tokenizer, vocab_json

    tokenizer = load_tokenizer(config.data.tokenizer)
    pieces = {int(key): piece for key, piece in vocab_json(tokenizer).items()}
    return pieces, BinaryBPELM.load(lm_path)


def build_decoder(checkpoint: Path, config, device: str, lm_path: Path):
    """Model from the checkpoint's own architecture, plus LM and CTC pieces."""
    import torch

    from training.model import build_model

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = dataclasses.replace(
        config.model.__class__(**payload["model_config"]), init=None, identity_heads=False
    )
    model = build_model(model_config)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    pieces, lm = decoder_assets(config, lm_path)
    return model, pieces, lm, payload


def decode_set(model, dataset, pieces, lm, *, device: str, max_seconds: float, max_clips: int):
    """Beam-decode one dev dataset; returns {id: hyp} and the elapsed seconds."""
    import torch
    from stt.data.dataset import collate, plan_batches
    from stt.decode import beam_search

    from training.model import forward

    batches = plan_batches(
        dataset.durations, max_seconds=max_seconds, max_clips=max_clips, seed=0, epoch=0
    )
    hyps: dict[str, str] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for index, positions in enumerate(batches, start=1):
            items = [dataset[position] for position in positions]
            mel, lengths, rows = collate(items)
            log_probs, out_lengths = forward(
                model, mel.to(device), lengths.to(device), checkpointing=False, training=False
            )
            probs = log_probs.detach().cpu()
            for row_index, row in enumerate(rows):
                length = int(out_lengths[row_index])
                hyps[row["id"]] = beam_search(
                    probs[row_index, :length],
                    length,
                    pieces,
                    beam_width=BEAM_WIDTH,
                    top_k=TOP_K,
                    lm=lm,
                    lm_weight=LM_WEIGHT,
                    token_bonus=TOKEN_BONUS,
                )
            if index % 25 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"    {index}/{len(batches)} batches, {len(hyps)} clips, {elapsed:.0f}s",
                    flush=True,
                )
    return hyps, time.perf_counter() - started


def decode_set_onnx(session, dataset, pieces, lm, *, max_seconds: float, max_clips: int):
    """Beam-decode one dev dataset through an ONNX session (batch 1)."""
    import numpy as np
    from stt.data.dataset import collate, plan_batches
    from stt.decode import beam_search

    batches = plan_batches(
        dataset.durations, max_seconds=max_seconds, max_clips=max_clips, seed=0, epoch=0
    )
    hyps: dict[str, str] = {}
    started = time.perf_counter()
    for index, positions in enumerate(batches, start=1):
        items = [dataset[position] for position in positions]
        mel, lengths, rows = collate(items)
        mels = mel.numpy()
        for row_index, row in enumerate(rows):
            length = int(lengths[row_index])
            batch = np.ascontiguousarray(
                mels[row_index : row_index + 1, :, :length], dtype=np.float32
            )
            log_probs = session.run(None, {"mel": batch})[0]
            frames = log_probs.shape[1]
            hyps[row["id"]] = beam_search(
                log_probs[0, :frames],
                frames,
                pieces,
                beam_width=BEAM_WIDTH,
                top_k=TOP_K,
                lm=lm,
                lm_weight=LM_WEIGHT,
                token_bonus=TOKEN_BONUS,
            )
        if index % 25 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"    {index}/{len(batches)} batches, {len(hyps)} clips, {elapsed:.0f}s",
                flush=True,
            )
    return hyps, time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="frozen_readout.py", description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="PyTorch checkpoint")
    parser.add_argument("--onnx", type=Path, help="INT8 ONNX graph (instead of --checkpoint)")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--hyp-out", type=Path, help="write hypotheses JSONL here")
    parser.add_argument("--reference-hyp", type=Path, default=DEFAULT_REFERENCE_HYP)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0, help="smoke: first N clips per set")
    parser.add_argument(
        "--lm", type=Path, default=None, help="decoder LM (default: the pinned 0.8.1 A3 LM)"
    )
    parser.add_argument(
        "--allow-other-lm",
        action="store_true",
        help="accept an LM that is not A3 (LM experiments only; the gate is then not comparable)",
    )
    args = parser.parse_args(argv)
    if bool(args.checkpoint) == bool(args.onnx):
        parser.error("pass exactly one of --checkpoint or --onnx")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    lm_info = resolve_lm(args.lm, allow_other=args.allow_other_lm)
    if not lm_info["is_a3"]:
        print(f"warning: decoding with a non-A3 LM ({lm_info['label']})", flush=True)

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(MOTH_STT))
    sys.path.insert(0, str(REPO / "vendor/moth-stt"))
    from moth_eval import macro_wer_bootstrap, score_rows

    from training.config import load_config
    from training.data import MultiMelDataset

    config = load_config(args.config)
    if args.onnx:
        import onnxruntime as ort

        session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        pieces, lm = decoder_assets(config, lm_info["path"])
        payload = {}
        print(
            f"onnx: {args.onnx} ({args.onnx.stat().st_size / 1e6:.2f} MB), "
            f"tokenizer {len(pieces)} pieces",
            flush=True,
        )
    else:
        model, pieces, lm, payload = build_decoder(
            args.checkpoint, config, args.device, lm_info["path"]
        )
        print(
            f"model: {sum(p.numel() for p in model.parameters()):,} params, "
            f"step {payload.get('global_step')}, tokenizer {len(pieces)} pieces",
            flush=True,
        )

    set_sources = {"it": config.data.dev[0], "en": config.data.dev[1]}
    results: dict[str, dict] = {}
    set_rows: dict[str, list[dict]] = {}
    all_hyps: dict[str, str] = {}
    for name, source in set_sources.items():
        dataset = MultiMelDataset(
            [source], augment=None, seed=0, limit=args.limit, role=f"dev-{name}"
        )
        print(f"{name}: {len(dataset)} clips, {sum(dataset.durations) / 60:.1f} min", flush=True)
        if args.onnx:
            hyps, elapsed = decode_set_onnx(
                session,
                dataset,
                pieces,
                lm,
                max_seconds=config.data.max_seconds,
                max_clips=config.data.max_clips,
            )
        else:
            hyps, elapsed = decode_set(
                model,
                dataset,
                pieces,
                lm,
                device=args.device,
                max_seconds=config.data.max_seconds,
                max_clips=config.data.max_clips,
            )
        scored = score_rows(dataset.rows, hyps, bootstrap=args.bootstrap, seed=0)
        results[name] = {
            "n": scored["n_utterances"],
            "wer": scored["overall"]["wer"],
            "wer_ci": scored["overall"]["wer_ci"],
            "seconds": round(elapsed, 1),
        }
        set_rows[name] = dataset.rows
        all_hyps.update(hyps)
        print(f"{name}: WER {scored['overall']['wer'] * 100:.4f} % ({elapsed:.0f}s)", flush=True)

    reference = {
        row["id"]: row["hyp"]
        for row in map(json.loads, args.reference_hyp.read_text(encoding="utf-8").splitlines())
    }
    ref_results: dict[str, dict] = {}
    for name, rows in set_rows.items():
        subset = {str(row["id"]): reference[str(row["id"])] for row in rows}
        ref_results[name] = score_rows(rows, subset, bootstrap=0, seed=0)
    ref_macro = macro_wer({name: ref_results[name]["overall"]["wer"] for name in ref_results})

    ours_macro = macro_wer({name: results[name]["wer"] for name in ("it", "en")})
    ours_rows = [row for name in ("it", "en") for row in set_rows[name]]
    ours_scored = score_rows(ours_rows, all_hyps, bootstrap=0, seed=0)
    reference_scored = {
        "per_utterance": {
            uid: entry
            for name in ("it", "en")
            for uid, entry in ref_results[name]["per_utterance"].items()
        }
    }
    paired = macro_wer_bootstrap(
        ours_rows, ours_scored, reference_scored, bootstrap=args.bootstrap, seed=0
    )
    verdict = gate_verdict(paired["point"], paired["ci"])

    output = {
        "tool": "tools/frozen_readout.py",
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
        "checkpoint_step": payload.get("global_step"),
        "onnx": str(args.onnx) if args.onnx else None,
        "onnx_sha256": sha256(args.onnx) if args.onnx else None,
        "decoder": {
            "mode": "beam",
            "beam_width": BEAM_WIDTH,
            "top_k": TOP_K,
            "lm_weight": LM_WEIGHT,
            "token_bonus": TOKEN_BONUS,
            "lm": str(lm_info["path"]),
            "lm_sha256": lm_info["sha256"],
            "lm_label": lm_info["label"],
            "lm_is_a3": lm_info["is_a3"],
        },
        "sets": results,
        "macro_wer": ours_macro,
        "reference": {
            "macro_wer": ref_macro,
            "sets": {name: ref_results[name]["overall"]["wer"] for name in ref_results},
            "hyp_sha256": sha256(args.reference_hyp),
            "reported_macro": REFERENCE_MACRO,
        },
        "paired_macro": paired,
        "gate": verdict,
        "elapsed_seconds": round(sum(results[name]["seconds"] for name in results), 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.hyp_out:
        args.hyp_out.write_text(
            "\n".join(
                json.dumps({"id": uid, "hyp": hyp}, ensure_ascii=False)
                for uid, hyp in all_hyps.items()
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "macro_wer": round(ours_macro * 100, 4),
                "reference": round(ref_macro * 100, 4),
                "diff_pp": round(paired["point"] * 100, 4),
                "ci_pp": [round(value * 100, 4) for value in paired["ci"]]
                if paired["ci"]
                else None,
                "gate": verdict,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
