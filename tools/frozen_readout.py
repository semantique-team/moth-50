"""Frozen E-002 readout for a MOTH-50 checkpoint or ONNX graph: beam 12 + A3 LM.

This is the gate decoder, not the trainer's greedy metric. Two modes share it:

- ``--mode macro`` (default): the dev-macro gate. Decodes MLS-IT and LS-EN dev
  (``config.data.dev[0]`` and ``[1]``) and runs the paired macro bootstrap against
  the retained 0.8.1 hypotheses behind the 23.0170 % reference.
- ``--mode mixed``: one mixed IT–EN set, normally S1-dev, given by ``--manifest``
  and ``--shards``. Reports pooled WER, ETA, insertions and the per-category
  table. With ``--reference-hyp`` it adds the paired WER/ETA bootstrap and the
  E-002 "no S1-dev regression outside its CI" verdict. The 0.8.1 reference
  hypotheses come from this same tool run with ``--onnx`` on the pinned 0.8.1
  graph (``tools/release_assets.py fetch moth-0.8.1-model``).

    python tools/release_assets.py fetch lm moth-0.8.1-model
    uv run --frozen --project ../moth-stt --extra training -- \
      python tools/frozen_readout.py \
      --checkpoint runs/E-005/base+6-16k/checkpoints/ckpt_best.pt \
      --config training/configs/base6-long.toml \
      --out runs/E-005/readout.json --device cpu

    # S1-dev: 0.8.1 reference first, then the candidate against it
    uv run --frozen --project ../moth-stt --extra training -- \
      python tools/frozen_readout.py --mode mixed \
      --onnx data/raw/moth-0.8.1/model.int8.onnx \
      --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl --shards data/shards/s1-dev \
      --config training/configs/base6-long.toml \
      --out runs/E-005/s1-dev-0.8.1.json --hyp-out runs/E-005/s1-dev-0.8.1.hyp.jsonl
    uv run --frozen --project ../moth-stt --extra training -- \
      python tools/frozen_readout.py --mode mixed \
      --onnx runs/E-005/release/model.int8.onnx \
      --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl --shards data/shards/s1-dev \
      --config training/configs/base6-long.toml \
      --reference-hyp runs/E-005/s1-dev-0.8.1.hyp.jsonl \
      --out runs/E-005/s1-dev-int8.json --hyp-out runs/E-005/s1-dev-int8.hyp.jsonl

Fixed decoder (experiments/E-002-protocol.md): beam 12, top-k 10, lm_weight
0.55, token_bonus 2.7 and the 0.8.1 A3 LM, resolved and sha256-checked by
``tools/release_assets.py`` (``data/raw/moth-0.8.1/lm_bpe3.bin``).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from collections.abc import Iterator
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


@dataclasses.dataclass(frozen=True)
class DecoderSettings:
    beam_width: int = BEAM_WIDTH
    top_k: int = TOP_K
    lm_weight: float = LM_WEIGHT
    token_bonus: float = TOKEN_BONUS


#: The E-002 protocol decoder; a gate readout never uses anything else.
FROZEN = DecoderSettings()


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


def no_regression_verdict(paired: dict) -> dict:
    """E-002 S1-dev rule on a ``moth_eval`` paired comparison (ours − reference).

    A regression is a worsening whose 95 % CI excludes zero: WER higher
    (``wer_diff_ci[0] > 0``) or ETA lower (``eta_diff_ci[1] < 0``). A tie is not a
    regression. Without a CI there is no verdict.
    """
    wer_ci = paired.get("wer_diff_ci")
    eta_ci = paired.get("eta_diff_ci")
    if wer_ci is None:
        return {"has_ci": False, "wer_regression": None, "eta_regression": None, "pass": None}
    wer_regression = wer_ci[0] > 0
    eta_regression = bool(eta_ci is not None and eta_ci[1] < 0)
    return {
        "has_ci": True,
        "wer_diff_pp": round(paired["wer_diff"] * 100, 4),
        "eta_diff_pp": (
            round(paired["eta_diff"] * 100, 4) if paired.get("eta_diff") is not None else None
        ),
        "wer_regression": wer_regression,
        "eta_regression": eta_regression,
        "pass": not (wer_regression or eta_regression),
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


def load_model(checkpoint: Path, config, device: str, weights: str = "auto"):
    """Model from the checkpoint's own architecture; returns ``(model, payload, weights)``.

    ``weights`` picks the raw or the EMA weights (``training/ema.py``); ``auto``
    takes the EMA weights when the checkpoint has them, since ``ckpt_best`` was
    selected on them.
    """
    import torch

    from training.ema import checkpoint_weights
    from training.model import build_model

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = dataclasses.replace(
        config.model.__class__(**payload["model_config"]), init=None, identity_heads=False
    )
    model = build_model(model_config)
    state, chosen = checkpoint_weights(payload, weights)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, payload, chosen


def iter_log_probs(
    model, dataset, *, device: str, max_seconds: float, max_clips: int
) -> Iterator[tuple[dict, object]]:
    """``(row, log_probs[T, V])`` per clip from the PyTorch model, in Moth's dev batches."""
    import torch
    from stt.data.dataset import collate, plan_batches

    from training.model import forward

    batches = plan_batches(
        dataset.durations, max_seconds=max_seconds, max_clips=max_clips, seed=0, epoch=0
    )
    with torch.inference_mode():
        for positions in batches:
            mel, lengths, rows = collate([dataset[position] for position in positions])
            log_probs, out_lengths = forward(
                model, mel.to(device), lengths.to(device), checkpointing=False, training=False
            )
            probs = log_probs.detach().cpu().numpy()
            for row_index, row in enumerate(rows):
                yield row, probs[row_index, : int(out_lengths[row_index])]


def iter_log_probs_onnx(
    session, dataset, *, max_seconds: float, max_clips: int
) -> Iterator[tuple[dict, object]]:
    """``(row, log_probs[T, V])`` per clip from an ONNX graph, batch 1 at the true length."""
    import numpy as np
    from stt.data.dataset import collate, plan_batches

    batches = plan_batches(
        dataset.durations, max_seconds=max_seconds, max_clips=max_clips, seed=0, epoch=0
    )
    for positions in batches:
        mel, lengths, rows = collate([dataset[position] for position in positions])
        mels = mel.numpy()
        for row_index, row in enumerate(rows):
            length = int(lengths[row_index])
            batch = np.ascontiguousarray(mels[row_index : row_index + 1, :, :length], np.float32)
            log_probs = session.run(None, {"mel": batch})[0]
            yield row, log_probs[0]


def beam_decode(log_probs, pieces, lm, settings: DecoderSettings) -> str:
    from stt.decode import beam_search

    return beam_search(
        log_probs,
        int(log_probs.shape[0]),
        pieces,
        beam_width=settings.beam_width,
        top_k=settings.top_k,
        lm=lm,
        lm_weight=settings.lm_weight,
        token_bonus=settings.token_bonus,
    )


def decode_set(clips, pieces, lm, settings: DecoderSettings, *, total: int):
    """Beam-decode ``(row, log_probs)`` pairs; returns ``{id: hyp}`` and the elapsed seconds."""
    hyps: dict[str, str] = {}
    started = time.perf_counter()
    for row, log_probs in clips:
        hyps[row["id"]] = beam_decode(log_probs, pieces, lm, settings)
        if len(hyps) % 250 == 0:
            elapsed = time.perf_counter() - started
            print(f"    {len(hyps)}/{total} clips, {elapsed:.0f}s", flush=True)
    return hyps, time.perf_counter() - started


def read_hyps(path: Path) -> dict[str, str]:
    hyps: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            hyps[str(row["id"])] = row["hyp"]
    return hyps


def write_hyps(path: Path, hyps: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps({"id": uid, "hyp": hyp}, ensure_ascii=False) for uid, hyp in hyps.items()
        )
        + "\n",
        encoding="utf-8",
    )


def mixed_summary(scored: dict) -> dict:
    """The reported block of a mixed-set ``score_rows`` result."""
    overall = scored["overall"]
    breakdown = scored.get("breakdown") or {}
    return {
        "n": scored["n_utterances"],
        "ref_words": overall["ref_words"],
        "wer": overall["wer"],
        "wer_ci": overall["wer_ci"],
        "eta": overall["eta"],
        "insertion_rate": overall["insertion_rate"],
        "eta_errors": overall["eta_errors"],
        "category": breakdown.get("category"),
        "speaker": breakdown.get("speaker"),
    }


def _pct(value: float | None) -> float | None:
    return None if value is None else round(value * 100, 4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="frozen_readout.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["macro", "mixed"], default="macro")
    parser.add_argument("--checkpoint", type=Path, help="PyTorch checkpoint")
    parser.add_argument("--onnx", type=Path, help="INT8 ONNX graph (instead of --checkpoint)")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--hyp-out", type=Path, help="write hypotheses JSONL here")
    parser.add_argument(
        "--reference-hyp",
        type=Path,
        default=None,
        help="reference hypotheses (macro default: the retained E-037 0.8.1 dev hypotheses)",
    )
    parser.add_argument("--manifest", type=Path, help="mixed mode: set manifest (S1-dev)")
    parser.add_argument("--shards", type=Path, help="mixed mode: mel shard directory")
    parser.add_argument("--name", default="s1-dev", help="mixed mode: set label")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument(
        "--weights",
        default="auto",
        choices=["auto", "model", "ema"],
        help="checkpoint weights: auto = EMA when present (training/ema.py)",
    )
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
    if args.mode == "mixed" and not (args.manifest and args.shards):
        parser.error("--mode mixed needs --manifest and --shards")
    if args.mode == "macro" and (args.manifest or args.shards):
        parser.error("--manifest/--shards belong to --mode mixed")
    if args.mode == "macro" and args.reference_hyp is None:
        args.reference_hyp = DEFAULT_REFERENCE_HYP
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    lm_info = resolve_lm(args.lm, allow_other=args.allow_other_lm)
    if not lm_info["is_a3"]:
        print(f"warning: decoding with a non-A3 LM ({lm_info['label']})", flush=True)

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(MOTH_STT))
    sys.path.insert(0, str(REPO / "vendor/moth-stt"))
    from moth_eval import _compare, macro_wer_bootstrap, score_rows

    from training.config import DataSource, load_config
    from training.data import MultiMelDataset

    config = load_config(args.config)
    pieces, lm = decoder_assets(config, lm_info["path"])
    if args.onnx:
        import onnxruntime as ort

        session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        payload: dict = {}
        weights = None
        print(
            f"onnx: {args.onnx} ({args.onnx.stat().st_size / 1e6:.2f} MB), "
            f"tokenizer {len(pieces)} pieces",
            flush=True,
        )
    else:
        model, payload, weights = load_model(args.checkpoint, config, args.device, args.weights)
        print(
            f"model: {sum(p.numel() for p in model.parameters()):,} params, "
            f"step {payload.get('global_step')}, {weights} weights, tokenizer {len(pieces)} pieces",
            flush=True,
        )

    def decode(dataset, name: str) -> tuple[dict[str, str], float]:
        print(f"{name}: {len(dataset)} clips, {sum(dataset.durations) / 60:.1f} min", flush=True)
        if args.onnx:
            clips = iter_log_probs_onnx(
                session,
                dataset,
                max_seconds=config.data.max_seconds,
                max_clips=config.data.max_clips,
            )
        else:
            clips = iter_log_probs(
                model,
                dataset,
                device=args.device,
                max_seconds=config.data.max_seconds,
                max_clips=config.data.max_clips,
            )
        return decode_set(clips, pieces, lm, FROZEN, total=len(dataset))

    output = {
        "tool": "tools/frozen_readout.py",
        "mode": args.mode,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
        "checkpoint_step": payload.get("global_step"),
        "checkpoint_weights": weights,
        "onnx": str(args.onnx) if args.onnx else None,
        "onnx_sha256": sha256(args.onnx) if args.onnx else None,
        "decoder": {
            "mode": "beam",
            **dataclasses.asdict(FROZEN),
            "lm": str(lm_info["path"]),
            "lm_sha256": lm_info["sha256"],
            "lm_label": lm_info["label"],
            "lm_is_a3": lm_info["is_a3"],
        },
    }

    if args.mode == "mixed":
        source = DataSource(manifest=args.manifest.resolve(), shards=args.shards.resolve())
        dataset = MultiMelDataset([source], augment=None, seed=0, limit=args.limit, role=args.name)
        all_hyps, elapsed = decode(dataset, args.name)
        scored = score_rows(dataset.rows, all_hyps, bootstrap=args.bootstrap, seed=0)
        summary = mixed_summary(scored)
        output |= {
            "set": {
                "name": args.name,
                "manifest": str(args.manifest),
                "manifest_sha256": sha256(args.manifest),
                "shards": str(args.shards),
                "limit": args.limit,
            },
            "metrics": summary,
            "reference": None,
            "paired": None,
            "no_regression": None,
            "elapsed_seconds": round(elapsed, 1),
        }
        if args.reference_hyp:
            reference = read_hyps(args.reference_hyp)
            ids = [str(row["id"]) for row in dataset.rows]
            missing = [uid for uid in ids if uid not in reference]
            if missing:
                raise SystemExit(
                    f"{args.reference_hyp} lacks {len(missing)} ids of this set "
                    f"(first: {missing[:3]}); decode the reference on the same manifest"
                )
            ref_scored = score_rows(
                dataset.rows, {uid: reference[uid] for uid in ids}, bootstrap=0, seed=0
            )
            ref_scored["hyp"] = str(args.reference_hyp)
            paired = _compare(scored, ref_scored, bootstrap=args.bootstrap, seed=0)
            verdict = no_regression_verdict(paired)
            verdict["comparable"] = lm_info["is_a3"]
            output |= {
                "reference": {
                    "hyp": str(args.reference_hyp),
                    "hyp_sha256": sha256(args.reference_hyp),
                    "wer": ref_scored["overall"]["wer"],
                    "eta": ref_scored["overall"]["eta"],
                    "category": (ref_scored.get("breakdown") or {}).get("category"),
                },
                "paired": paired,
                "no_regression": verdict,
            }
        printed = {
            "set": args.name,
            "wer": _pct(summary["wer"]),
            "wer_ci": [_pct(value) for value in summary["wer_ci"]] if summary["wer_ci"] else None,
            "eta": _pct(summary["eta"]),
            "no_regression": output["no_regression"],
        }
    else:
        set_sources = {"it": config.data.dev[0], "en": config.data.dev[1]}
        results: dict[str, dict] = {}
        set_rows: dict[str, list[dict]] = {}
        all_hyps: dict[str, str] = {}
        for dataset_name, source in set_sources.items():
            dataset = MultiMelDataset(
                [source], augment=None, seed=0, limit=args.limit, role=f"dev-{dataset_name}"
            )
            hyps, elapsed = decode(dataset, dataset_name)
            scored = score_rows(dataset.rows, hyps, bootstrap=args.bootstrap, seed=0)
            results[dataset_name] = {
                "n": scored["n_utterances"],
                "wer": scored["overall"]["wer"],
                "wer_ci": scored["overall"]["wer_ci"],
                "seconds": round(elapsed, 1),
            }
            set_rows[dataset_name] = dataset.rows
            all_hyps.update(hyps)
            print(f"{dataset_name}: WER {_pct(scored['overall']['wer'])} % ({elapsed:.0f}s)")

        reference = read_hyps(args.reference_hyp)
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
        verdict["comparable"] = lm_info["is_a3"]
        output |= {
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
        printed = {
            "macro_wer": _pct(ours_macro),
            "reference": _pct(ref_macro),
            "diff_pp": _pct(paired["point"]),
            "ci_pp": [_pct(value) for value in paired["ci"]] if paired["ci"] else None,
            "gate": verdict,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.hyp_out:
        write_hyps(args.hyp_out, all_hyps)
    print(json.dumps(printed, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
