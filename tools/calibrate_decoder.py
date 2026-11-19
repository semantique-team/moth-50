"""E-009: recalibrate the beam decoder (lm_weight × token_bonus) on dev, never on T.

The frozen settings (lm_weight 0.55, token_bonus 2.7) were calibrated by Moth E-022
for the 11.6 M tiny+2 model; a stronger acoustic model usually wants a lighter LM
and a different length bonus. This tool runs the acoustic model once per clip,
beam-decodes every grid point from the same log-probs (in worker processes if
``--workers`` > 1), scores dev-macro (MLS-IT + LS-EN) and applies the amended
E-009 rule: rank on **dev-macro only**.

S1-dev is **not** a grid input. The selected point gets one veto readout
(`tools/frozen_readout.py --experiment-decoder --lm-weight … --token-bonus …`)
against the frozen MOTH-50 hypotheses; a significant WER or ETA regression keeps
the frozen setting. ``--select-on s1`` is the old path and needs an explicit
``--allow-s1-selection``. Beam width, top-k and the A3 LM stay fixed.

    uv run --frozen --project ../moth-stt --extra training -- \
      python tools/calibrate_decoder.py \
      --onnx runs/E-005/release-a3/model.int8.onnx \
      --config training/configs/base6-long.toml \
      --out runs/E-009/grid.json --workers 8

T-600, FLEURS and CS-FLEURS are test sets: never pass them here.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from frozen_readout import (
    FROZEN,
    MOTH_STT,
    REPO,
    DecoderSettings,
    beam_decode,
    decoder_assets,
    iter_log_probs,
    iter_log_probs_onnx,
    load_model,
    macro_wer,
    sha256,
)
from release_assets import resolve_lm

DEFAULT_LM_WEIGHTS = "0.25,0.35,0.45,0.55,0.65"
DEFAULT_TOKEN_BONUSES = "1.5,2.0,2.7,3.5"


def parse_values(text: str) -> list[float]:
    values = [float(part) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("empty grid")
    return values


def grid_settings(lm_weights: list[float], token_bonuses: list[float]) -> list[DecoderSettings]:
    """Every (lm_weight, token_bonus) pair; the frozen point is always included."""
    settings = [
        dataclasses.replace(FROZEN, lm_weight=weight, token_bonus=bonus)
        for weight in lm_weights
        for bonus in token_bonuses
    ]
    if FROZEN not in settings:
        settings.append(FROZEN)
    return settings


def _significantly_worse(ci: list[float] | None) -> bool:
    """``ci`` of (candidate − frozen) for an error rate: worse when it excludes zero above."""
    return bool(ci is not None and ci[0] > 0)


def _significantly_better(ci: list[float] | None) -> bool:
    return bool(ci is not None and ci[1] < 0)


def select_setting(points: list[dict], *, objective: str = "macro") -> dict:
    """The pre-registered E-009 rule.

    ``objective="macro"`` (the amended default) ranks on dev-macro only; S1-dev
    is a separate one-point veto, never a grid ranking. ``objective="s1"`` is the
    old path and needs an explicit opt-in at the CLI.

    1. Drop points that are significantly worse than the frozen point on dev-macro
       WER, or (``objective="s1"``) on S1-dev ETA, paired 95 % CI excluding zero.
    2. With ``objective="s1"``, pick the lowest S1-dev WER (ties: higher ETA, lower
       dev-macro, closer to frozen); otherwise the lowest dev-macro WER.
    3. Adopt it only if it beats the frozen point on that objective with a paired
       CI excluding zero; otherwise keep the frozen settings (a tie is a tie).
    """
    if objective not in ("macro", "s1"):
        raise ValueError(f"objective must be 'macro' or 's1', got {objective!r}")
    frozen = next(point for point in points if point["frozen"])
    with_s1 = objective == "s1"

    def eligible(point: dict) -> bool:
        if point["frozen"]:
            return True
        macro = point.get("macro_vs_frozen")
        if macro is not None and _significantly_worse(macro["ci"]):
            return False
        if with_s1:
            s1 = point.get("s1")
            if s1 is not None:
                eta_ci = s1["vs_frozen"].get("eta_diff_ci")
                if eta_ci is not None and eta_ci[1] < 0:
                    return False
        return True

    def distance(point: dict) -> tuple[float, float]:
        return (
            abs(point["lm_weight"] - FROZEN.lm_weight),
            abs(point["token_bonus"] - FROZEN.token_bonus),
        )

    def key(point: dict):
        macro = point["macro_wer"] if point["macro_wer"] is not None else 0.0
        if with_s1:
            eta = point["s1"]["eta"] if point["s1"]["eta"] is not None else 0.0
            return (point["s1"]["wer"], -eta, macro, distance(point))
        return (macro, distance(point))

    candidates = [point for point in points if eligible(point)]
    best = min(candidates, key=key)
    if best["frozen"]:
        better = False
    elif with_s1:
        better = _significantly_better(best["s1"]["vs_frozen"]["wer_diff_ci"])
    else:
        better = _significantly_better(best["macro_vs_frozen"]["ci"])
    chosen = best if better else frozen
    return {
        "objective": "s1-dev WER" if with_s1 else "dev-macro WER",
        "eligible": len(candidates),
        "best": {"lm_weight": best["lm_weight"], "token_bonus": best["token_bonus"]},
        "adopt": better,
        "selected": {"lm_weight": chosen["lm_weight"], "token_bonus": chosen["token_bonus"]},
    }


_WORKER: dict = {}


def _init_worker(paths: list[str], lm_path: str, pieces: dict, settings: list) -> None:
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    from stt.lm import BinaryBPELM

    _WORKER.update(pieces=pieces, lm=BinaryBPELM.load(lm_path), settings=settings)


def _decode_clip(uid: str, log_probs) -> tuple[str, list[str]]:
    return uid, [
        beam_decode(log_probs, _WORKER["pieces"], _WORKER["lm"], setting)
        for setting in _WORKER["settings"]
    ]


def decode_grid(clips, pieces, lm, settings, *, workers: int, lm_path: Path, total: int):
    """``[{id: hyp}]`` per setting, from one acoustic pass over ``clips``."""
    results: list[dict[str, str]] = [{} for _ in settings]
    started = time.perf_counter()
    done_count = 0

    def collect(uid: str, hyps: list[str]) -> None:
        nonlocal done_count
        for index, hyp in enumerate(hyps):
            results[index][uid] = hyp
        done_count += 1
        if done_count % 100 == 0:
            elapsed = time.perf_counter() - started
            print(f"    {done_count}/{total} clips × {len(settings)} settings, {elapsed:.0f}s")

    if workers <= 1:
        for row, log_probs in clips:
            collect(row["id"], [beam_decode(log_probs, pieces, lm, s) for s in settings])
        return results

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(list(sys.path), str(lm_path), pieces, settings),
    ) as pool:
        pending = set()
        for row, log_probs in clips:
            pending.add(pool.submit(_decode_clip, row["id"], log_probs))
            if len(pending) >= workers * 4:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    collect(*future.result())
        for future in wait(pending)[0]:
            collect(*future.result())
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="calibrate_decoder.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lm-weights", default=DEFAULT_LM_WEIGHTS)
    parser.add_argument("--token-bonuses", default=DEFAULT_TOKEN_BONUSES)
    parser.add_argument("--s1-dev-manifest", type=Path)
    parser.add_argument("--s1-dev-shards", type=Path)
    parser.add_argument(
        "--select-on",
        choices=["macro", "s1"],
        default="macro",
        help="E-009 objective: macro (amended default) or s1 (old path, explicit opt-in)",
    )
    parser.add_argument(
        "--allow-s1-selection",
        action="store_true",
        help="required for --select-on s1 (uses the product dev set for ranking)",
    )
    parser.add_argument("--no-macro", action="store_true", help="skip MLS-IT + LS-EN dev")
    parser.add_argument("--limit", type=int, default=0, help="first N clips per set")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--weights", default="auto", choices=["auto", "model", "ema"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--lm", type=Path, default=None)
    parser.add_argument("--allow-other-lm", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.checkpoint) == bool(args.onnx):
        parser.error("pass exactly one of --checkpoint or --onnx")
    if bool(args.s1_dev_manifest) != bool(args.s1_dev_shards):
        parser.error("--s1-dev-manifest and --s1-dev-shards go together")
    if args.select_on == "s1" and not args.allow_s1_selection:
        parser.error("--select-on s1 needs --allow-s1-selection (it ranks on S1-dev)")
    if args.select_on == "s1" and not args.s1_dev_manifest:
        parser.error("--select-on s1 needs --s1-dev-manifest and --s1-dev-shards")
    if args.select_on == "macro" and args.s1_dev_manifest:
        parser.error(
            "the S1-dev veto is a separate one-point readout; do not pass S1-dev to the grid"
        )
    if args.no_macro and args.select_on != "s1":
        parser.error("--no-macro is only for the --select-on s1 path")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    settings = grid_settings(parse_values(args.lm_weights), parse_values(args.token_bonuses))
    frozen_index = settings.index(FROZEN)
    lm_info = resolve_lm(args.lm, allow_other=args.allow_other_lm)

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
    else:
        model, payload, weights = load_model(args.checkpoint, config, args.device, args.weights)

    sets: dict[str, DataSource] = {}
    if not args.no_macro:
        sets |= {"it": config.data.dev[0], "en": config.data.dev[1]}
    if args.s1_dev_manifest:
        sets["s1"] = DataSource(
            manifest=args.s1_dev_manifest.resolve(), shards=args.s1_dev_shards.resolve()
        )

    started = time.perf_counter()
    rows: dict[str, list[dict]] = {}
    hyps: dict[str, list[dict[str, str]]] = {}
    for name, source in sets.items():
        dataset = MultiMelDataset([source], augment=None, seed=0, limit=args.limit, role=name)
        print(f"{name}: {len(dataset)} clips × {len(settings)} settings", flush=True)
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
        rows[name] = dataset.rows
        hyps[name] = decode_grid(
            clips,
            pieces,
            lm,
            settings,
            workers=args.workers,
            lm_path=lm_info["path"],
            total=len(dataset),
        )

    scored = {
        name: [score_rows(rows[name], grid, bootstrap=0, seed=0) for grid in hyps[name]]
        for name in sets
    }

    def merged_macro(position: int) -> dict:
        return {
            "per_utterance": scored["it"][position]["per_utterance"]
            | scored["en"][position]["per_utterance"]
        }

    points = []
    for index, setting in enumerate(settings):
        point = {
            "lm_weight": setting.lm_weight,
            "token_bonus": setting.token_bonus,
            "frozen": index == frozen_index,
            "sets": {name: scored[name][index]["overall"]["wer"] for name in sets},
            "macro_wer": None,
            "macro_vs_frozen": None,
            "s1": None,
        }
        if not args.no_macro:
            point["macro_wer"] = macro_wer({"it": point["sets"]["it"], "en": point["sets"]["en"]})
            paired = macro_wer_bootstrap(
                rows["it"] + rows["en"],
                merged_macro(index),
                merged_macro(frozen_index),
                bootstrap=args.bootstrap,
                seed=0,
            )
            point["macro_vs_frozen"] = {"point": paired["point"], "ci": paired["ci"]}
        if "s1" in sets:
            ours = scored["s1"][index]
            compare = _compare(ours, scored["s1"][frozen_index], bootstrap=args.bootstrap, seed=0)
            point["s1"] = {
                "wer": ours["overall"]["wer"],
                "eta": ours["overall"]["eta"],
                "insertion_rate": ours["overall"]["insertion_rate"],
                "vs_frozen": {
                    key: compare[key]
                    for key in ("wer_diff", "wer_diff_ci", "eta_diff", "eta_diff_ci")
                },
            }
        points.append(point)

    selection = select_setting(points, objective=args.select_on)
    output = {
        "tool": "tools/calibrate_decoder.py",
        "experiment": "E-009",
        "objective": args.select_on,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_step": payload.get("global_step"),
        "checkpoint_weights": weights,
        "onnx": str(args.onnx) if args.onnx else None,
        "onnx_sha256": sha256(args.onnx) if args.onnx else None,
        "fixed": {"beam_width": FROZEN.beam_width, "top_k": FROZEN.top_k},
        "lm": {
            "path": str(lm_info["path"]),
            "sha256": lm_info["sha256"],
            "is_a3": lm_info["is_a3"],
        },
        "sets": {
            name: {"manifest": str(source.manifest), "n": len(rows[name])}
            for name, source in sets.items()
        },
        "limit": args.limit,
        "bootstrap": args.bootstrap,
        "points": points,
        "selection": selection,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    table = sorted(points, key=lambda p: (p["lm_weight"], p["token_bonus"]))
    for point in table:
        macro = f"{point['macro_wer'] * 100:7.3f}" if point["macro_wer"] is not None else "      -"
        s1 = (
            f"{point['s1']['wer'] * 100:7.3f} / ETA {point['s1']['eta'] * 100:6.2f}"
            if point["s1"] and point["s1"]["eta"] is not None
            else "-"
        )
        mark = "*" if point["frozen"] else " "
        print(f"{mark} lm {point['lm_weight']:.2f} bonus {point['token_bonus']:.2f}  {macro}  {s1}")
    print(json.dumps(selection, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
