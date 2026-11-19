"""MOTH-50 trainer: base+6 CTC on MPS/CPU with activation checkpointing.

This is the first deliberate port of the Moth training path, not a copy of
``stt/train.py``. It reuses Moth's model, tokenizer, mel cache, collate,
duration-batched planner and evaluator, while owning the loop and the settings
that differ for MOTH-50:

- ``EncoderConfig`` comes from the TOML config (``base`` by default), never from
  the Whisper-tiny default;
- per-block activation checkpointing is the default (D15);
- the 192 audio-s / 32-clip batch cap is explicit;
- checkpoints, resume, JSONL logs, MPS memory and a greedy dev readout are
  built in;
- the final beam+LM evaluation remains the frozen E-002 protocol, not this
  monitoring metric.

Run from the repository root with the Moth training environment:

    uv run --frozen --project ../moth-stt --extra training -- \
      python -m training.train --config training/configs/base6.toml \
      --moth-stt ../moth-stt --out runs/E-003/base6 --device mps

The data sources must be built into mel shards first:

    python -m stt.data.build_mel --manifest ... --audio-root ... --out-dir ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.config import TrainConfig, load_config, override_seed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _optimizer_to(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _lr_factor(step: int, warmup: int, total: int, min_factor: float) -> float:
    if warmup and step < warmup:
        return max(1e-8, step / warmup)
    if total <= warmup:
        return 1.0
    progress = min(1.0, max(0.0, (step - warmup) / (total - warmup)))
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _set_lr(optimizer, base_lrs: list[float], factor: float) -> None:
    for group, base in zip(optimizer.param_groups, base_lrs, strict=True):
        group["lr"] = base * factor


def _autocast(device: torch.device, mode: str):
    if mode == "off":
        return nullcontext()
    dtype = torch.bfloat16 if mode == "bf16" else torch.float16
    return torch.autocast(device.type, dtype=dtype)


def _encode_rows(rows: list[dict], tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    from stt.tokenizer import encode

    sequences = [encode(tokenizer, row["norm"]) for row in rows]
    flat = [token for sequence in sequences for token in sequence]
    if not flat:
        raise ValueError("empty CTC targets")
    return (
        torch.tensor(flat, dtype=torch.long),
        torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long),
    )


def _save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    config: TrainConfig,
    epoch: int,
    batch_index: int,
    global_step: int,
    total_steps: int,
    best_dev_wer: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": _to_cpu(model.state_dict()),
            "optimizer": _to_cpu(optimizer.state_dict()),
            "config_text": config.path.read_text(encoding="utf-8"),
            "model_config": asdict(config.model),
            "epoch": epoch,
            "batch_index": batch_index,
            "global_step": global_step,
            "total_steps": total_steps,
            "best_dev_wer": best_dev_wer,
        },
        path,
    )


def _load_checkpoint(path: Path, *, model, optimizer, device: torch.device) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    _optimizer_to(optimizer, device)
    return payload


def _prepare_batch(dataset, positions: list[int], tokenizer, device: torch.device):
    from stt.data.dataset import collate

    items = [dataset[position] for position in positions]
    mel, lengths, rows = collate(items)
    targets, target_lengths = _encode_rows(rows, tokenizer)
    audio_seconds = sum(float(row["duration"]) for row in rows)
    return (
        mel.to(device),
        lengths.to(device),
        targets.to(device),
        target_lengths.to(device),
        rows,
        audio_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--moth-stt", type=Path, required=True, help="Moth checkout with stt/ (not release/)"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=0, help="override train.max_steps")
    parser.add_argument(
        "--head-blocks", type=int, default=None, help="override model.head_blocks (E-003 arms)"
    )
    parser.add_argument("--seed", type=int, default=None, help="override train.seed and data.seed")
    parser.add_argument("--dry-run", action="store_true", help="build model/data and exit")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.moth_stt.resolve()))
    import torch.nn.functional as F
    from stt.data.dataset import plan_batches
    from stt.tokenizer import load_tokenizer

    from training.data import load_datasets
    from training.evaluate import evaluate
    from training.model import (
        build_model,
        empty_cache,
        forward,
        memory_gb,
        synchronize,
    )

    config = load_config(args.config)
    if args.head_blocks is not None:
        config = replace(config, model=replace(config.model, head_blocks=args.head_blocks))
    if args.seed is not None:
        config = override_seed(config, args.seed)
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    torch.manual_seed(config.seed)

    tokenizer = load_tokenizer(config.data.tokenizer)
    train_data, dev_data = load_datasets(config.data)
    model = build_model(config.model).to(device)
    params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"model: {params:,} params, encoder={config.model.encoder()}, "
        f"head_blocks={config.model.head_blocks}, checkpointing={config.model.checkpointing}"
    )
    print(f"data: {len(train_data)} train clips, {len(dev_data) if dev_data else 0} dev clips")
    if args.dry_run:
        print(json.dumps({"params": params, "train": len(train_data)}, indent=2))
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.toml").write_text(config.path.read_text(encoding="utf-8"), encoding="utf-8")
    run_meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": args.device,
        "seed": config.seed,
        "params": params,
        "encoder": config.model.encoder(),
        "head_blocks": config.model.head_blocks,
        "checkpointing": config.model.checkpointing,
        "tokenizer": str(config.data.tokenizer),
        "tokenizer_sha256": _sha256(config.data.tokenizer),
        "train_clips": len(train_data),
        "dev_clips": len(dev_data) if dev_data else 0,
    }
    (args.out / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": config.optimizer.lr_encoder},
            {
                "params": [*model.head_blocks.parameters(), *model.head.parameters()],
                "lr": config.optimizer.lr_head,
            },
        ],
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    global_step = 0
    start_epoch = 0
    start_batch = 0
    max_steps = args.max_steps or config.max_steps
    total_steps = max_steps
    best_dev_wer: float | None = None
    if args.resume:
        payload = _load_checkpoint(args.resume, model=model, optimizer=optimizer, device=device)
        global_step = int(payload["global_step"])
        start_epoch = int(payload["epoch"])
        start_batch = int(payload["batch_index"])
        total_steps = total_steps or int(payload.get("total_steps") or 0)
        best_dev_wer = payload.get("best_dev_wer")
        print(
            f"resumed {args.resume} at step {global_step}, epoch {start_epoch}, batch {start_batch}"
        )

    log_path = args.out / "train.log.jsonl"
    log_handle = log_path.open("a", encoding="utf-8")
    epoch_batches: list[list[int]] = []
    peak_memory = 0.0
    nonfinite = 0
    last_loss = float("nan")
    started = time.perf_counter()

    try:
        for epoch in range(start_epoch, config.epochs):
            train_data.set_epoch(epoch)
            epoch_batches = plan_batches(
                train_data.durations,
                max_seconds=config.data.max_seconds,
                max_clips=config.data.max_clips,
                seed=config.data.seed,
                epoch=epoch,
            )
            if not total_steps:
                total_steps = len(epoch_batches) * config.epochs
            batches = epoch_batches[start_batch:] if epoch == start_epoch else epoch_batches
            if not batches and epoch == start_epoch:
                continue
            model.train()
            for batch_offset, positions in enumerate(batches, start=start_batch):
                global_step += 1
                factor = _lr_factor(
                    global_step,
                    config.schedule.warmup_steps,
                    total_steps,
                    config.schedule.min_factor,
                )
                _set_lr(optimizer, base_lrs, factor)
                (
                    mel,
                    lengths,
                    targets,
                    target_lengths,
                    _rows,
                    audio_seconds,
                ) = _prepare_batch(train_data, positions, tokenizer, device)
                optimizer.zero_grad(set_to_none=True)
                synchronize(device)
                step_started = time.perf_counter()
                with _autocast(device, config.autocast):
                    log_probs, out_lengths = forward(
                        model,
                        mel,
                        lengths,
                        checkpointing=config.model.checkpointing,
                        training=True,
                    )
                    loss = F.ctc_loss(
                        log_probs.transpose(0, 1),
                        targets,
                        out_lengths,
                        target_lengths,
                        blank=0,
                        zero_infinity=True,
                    )
                if bool(torch.isfinite(loss)):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip)
                    optimizer.step()
                    last_loss = float(loss.detach())
                else:
                    nonfinite += 1
                synchronize(device)
                step_seconds = time.perf_counter() - step_started
                reading = memory_gb(device)
                peak_memory = max(peak_memory, reading)
                if config.empty_cache_every and global_step % config.empty_cache_every == 0:
                    empty_cache(device)
                if global_step % config.log_every == 0:
                    entry = {
                        "step": global_step,
                        "epoch": epoch,
                        "batch": batch_offset,
                        "loss": round(last_loss, 5),
                        "nonfinite_steps": nonfinite,
                        "lr_head": optimizer.param_groups[1]["lr"],
                        "lr_encoder": optimizer.param_groups[0]["lr"],
                        "memory_gb": round(reading, 3),
                        "step_seconds": round(step_seconds, 4),
                        "audio_seconds": round(audio_seconds, 2),
                        "audio_s_per_s": round(audio_seconds / step_seconds, 2),
                    }
                    log_handle.write(json.dumps(entry) + "\n")
                    log_handle.flush()
                    print(json.dumps(entry), flush=True)
                if (
                    dev_data is not None
                    and config.eval_every_steps
                    and global_step % config.eval_every_steps == 0
                ):
                    print(f"evaluating dev at step {global_step}", flush=True)
                    metrics = evaluate(
                        model,
                        dev_data,
                        tokenizer=tokenizer,
                        device=device,
                        max_seconds=config.data.eval_max_seconds,
                        max_clips=config.data.eval_max_clips,
                        autocast=config.autocast,
                    )
                    metrics_path = args.out / f"dev-{global_step:07d}.json"
                    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
                    print(f"dev step {global_step}: {metrics}", flush=True)
                    model.train()
                    if best_dev_wer is None or metrics["wer"] < best_dev_wer:
                        best_dev_wer = metrics["wer"]
                        _save_checkpoint(
                            args.out / "checkpoints" / "ckpt_best.pt",
                            model=model,
                            optimizer=optimizer,
                            config=config,
                            epoch=epoch,
                            batch_index=batch_offset + 1,
                            global_step=global_step,
                            total_steps=total_steps,
                            best_dev_wer=best_dev_wer,
                        )
                if (
                    config.checkpoint_every_steps
                    and global_step % config.checkpoint_every_steps == 0
                ):
                    _save_checkpoint(
                        args.out / "checkpoints" / "ckpt_last.pt",
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        epoch=epoch,
                        batch_index=batch_offset + 1,
                        global_step=global_step,
                        total_steps=total_steps,
                        best_dev_wer=best_dev_wer,
                    )
                if max_steps and global_step >= max_steps:
                    break
            _save_checkpoint(
                args.out / "checkpoints" / "ckpt_last.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                batch_index=len(epoch_batches),
                global_step=global_step,
                total_steps=total_steps,
                best_dev_wer=best_dev_wer,
            )
            if max_steps and global_step >= max_steps:
                break
    finally:
        log_handle.close()

    summary = {
        "steps": global_step,
        "epochs_configured": config.epochs,
        "total_steps": total_steps,
        "last_loss": None if math.isnan(last_loss) else round(last_loss, 5),
        "nonfinite_steps": nonfinite,
        "best_dev_wer": best_dev_wer,
        "peak_memory_gb": round(peak_memory, 3),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "checkpoint_last": str(args.out / "checkpoints" / "ckpt_last.pt"),
        "checkpoint_best": str(args.out / "checkpoints" / "ckpt_best.pt"),
    }
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
