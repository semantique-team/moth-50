#!/usr/bin/env python3
"""Preflight E-003 without training.

Checks that every configured manifest and mel shard exists, counts hours and
rows, runs Moth's train-row guard, verifies the pinned tokenizer, and prints the
four architecture arms with their parameter/package budgets. Missing data is
reported, not treated as a crash, so this can be run while downloads are still
in progress.

Usage:
    uv run --frozen --project ../moth-stt --extra training -- \
      python -m training.preflight --config training/configs/base6.toml \
      --moth-stt ../moth-stt --out runs/E-003/preflight.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.architecture_budget import Candidate, acoustic_params, estimate
from training.config import DataSource, load_config

ARMS = (0, 2, 4, 6)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index_rows(shards: Path) -> dict | None:
    path = shards / "index.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _check_source(source: DataSource, role: str) -> dict:
    result = {
        "role": role,
        "manifest": str(source.manifest),
        "shards": str(source.shards),
        "repeat": source.repeat,
        "manifest_ok": source.manifest.is_file(),
        "shards_ok": (source.shards / "index.json").is_file(),
    }
    if not result["manifest_ok"]:
        result["status"] = "missing manifest"
        return result
    if not result["shards_ok"]:
        result["status"] = "missing mel shards"
        return result
    try:
        from stt.manifest import load_manifest

        rows = load_manifest(source.manifest)
    except Exception as error:  # noqa: BLE001 - report the exact preflight failure
        result["status"] = f"invalid manifest: {error}"
        return result
    index = _index_rows(source.shards) or {}
    missing = [row["id"] for row in rows if row["id"] not in index]
    result.update(
        {
            "rows": len(rows),
            "hours": round(sum(float(row.get("duration", 0.0)) for row in rows) / 3600, 3),
            "index_entries": len(index),
            "missing_from_index": len(missing),
            "status": "ready" if not missing else f"missing {len(missing)} rows from index",
        }
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--moth-stt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="exit 1 unless every source is ready")
    parser.add_argument("--steps", type=int, default=4000, help="planned steps per arm")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.moth_stt.resolve()))
    config = load_config(args.config)

    train = [_check_source(source, "train") for source in config.data.train]
    dev = [_check_source(source, "dev") for source in config.data.dev]

    # Run Moth's training-row guard only when the manifests are readable.
    guard = "skipped"
    if all(item["manifest_ok"] for item in train + dev):
        try:
            from stt.manifest import load_manifest, validate_training_rows

            train_rows = []
            for source in config.data.train:
                rows = load_manifest(source.manifest)
                train_rows.extend(rows * source.repeat)
            dev_rows = []
            for source in config.data.dev:
                dev_rows.extend(load_manifest(source.manifest))
            validate_training_rows(train_rows, dev_rows)
            guard = "ok"
        except Exception as error:  # noqa: BLE001 - report the exact guard failure
            guard = f"failed: {error}"

    tokenizer = {
        "path": str(config.data.tokenizer),
        "exists": config.data.tokenizer.is_file(),
        "sha256": _sha256(config.data.tokenizer) if config.data.tokenizer.is_file() else None,
    }
    arms = []
    for extra in ARMS:
        candidate = Candidate(f"base+{extra}", 512, 6, 2048, extra=extra)
        estimate_row = estimate(candidate)
        arms.append(
            {
                "arm": f"base+{extra}",
                "head_blocks": extra,
                "params": acoustic_params(candidate),
                "int8_mb": round(estimate_row["int8_mb"], 2),
                "package_mb": round(estimate_row["package_mb"], 2),
                "headroom_mb": round(estimate_row["headroom_mb"], 2),
                "fits": bool(estimate_row["fits"]),
                "hours_per_epoch": round(estimate_row["hours_per_epoch"], 2),
                "hours_per_epoch_checkpointing": round(
                    estimate_row["hours_per_epoch_checkpointing"], 2
                ),
                "planned_step_hours": round(
                    args.steps
                    * estimate_row["hours_per_epoch_checkpointing"]
                    * 3600
                    / 13_283
                    / 3600,
                    2,
                ),
            }
        )

    probe = ROOT / "runs" / "E-002" / "train-probe-checkpointing.json"
    memory = None
    if probe.is_file():
        payload = json.loads(probe.read_text(encoding="utf-8"))
        memory = {
            "probe": str(probe),
            "checkpointing": payload.get("checkpointing"),
            "max_observed_gb": payload.get("memory_gb", {}).get("max_observed"),
            "median_step_s": payload.get("timing_s", {}).get("median"),
        }

    ready = all(item["status"] == "ready" for item in train + dev)
    report = {
        "tool": "training/preflight.py",
        "config": str(config.path),
        "ready": ready,
        "strict": args.strict,
        "train_guard": guard,
        "tokenizer": tokenizer,
        "model": {
            "preset": config.model.preset,
            "encoder": config.model.encoder(),
            "checkpointing": config.model.checkpointing,
        },
        "data": {"train": train, "dev": dev},
        "arms": arms,
        "memory_probe": memory,
        "planned_steps_per_arm": args.steps,
        "commands": [
            (
                "uv run --frozen --project ../moth-stt --extra training -- "
                f"python -m training.train --config {config.path} "
                f"--head-blocks {extra} --max-steps {args.steps} "
                f"--out runs/E-003/base+{extra} --device mps"
            )
            for extra in ARMS
        ],
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"ready: {ready}  train_guard: {guard}  tokenizer_sha: {tokenizer['sha256']}")
    for section in ("train", "dev"):
        for item in report["data"][section]:
            print(
                f"  {section:5s} {item['status']:34s} "
                f"{item.get('rows', '-'):>7} rows {item.get('hours', '-'):>7} h  "
                f"{item['manifest']}"
            )
    for arm in arms:
        print(
            f"  {arm['arm']:7s} {arm['params']:>10,} params  "
            f"{arm['package_mb']:6.2f} MB  room {arm['headroom_mb']:5.2f} MB  "
            f"{arm['hours_per_epoch_checkpointing']:4.1f} h/epoch ckpt"
        )
    if memory:
        print(
            f"  E-002 memory: {memory['max_observed_gb']} GB, "
            f"{memory['median_step_s']} s/step (checkpointing={memory['checkpointing']})"
        )
    print("commands:")
    for command in report["commands"]:
        print(f"  {command}")
    if args.strict and not ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
