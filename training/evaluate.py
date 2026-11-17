"""Greedy dev evaluation for the MOTH-50 trainer.

This is a monitoring metric, not the frozen product protocol: it uses greedy CTC
decoding without the A3 LM. The E-003 gate still uses the beam+LM readout in
``experiments/E-002-protocol.md``. The greedy metric is cheap enough to run
inside the training loop and catches a broken model early.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext

import torch
from stt.data.dataset import collate, plan_batches
from stt.eval import score_rows
from stt.tokenizer import decode

from training.model import forward


def _decode_batch(
    log_probs: torch.Tensor, out_lengths: torch.Tensor, decode_text: Callable[[list[int]], str]
) -> list[str]:
    best = log_probs.argmax(dim=-1)
    hypotheses = []
    for row in range(best.shape[0]):
        ids: list[int] = []
        previous = -1
        for token in best[row, : int(out_lengths[row])].tolist():
            if token != 0 and token != previous:
                ids.append(int(token))
            previous = int(token)
        hypotheses.append(decode_text(ids))
    return hypotheses


def evaluate(
    model,
    dataset,
    *,
    tokenizer,
    device: torch.device,
    max_seconds: float,
    max_clips: int,
    autocast: str,
    name: str = "dev",
) -> dict:
    """Greedy WER/ETA on a ``MultiMelDataset``; returns the core metrics."""
    model.eval()
    batches = plan_batches(
        dataset.durations, max_seconds=max_seconds, max_clips=max_clips, seed=0, epoch=0
    )
    hyps: dict[str, str] = {}
    context = (
        torch.autocast(device.type, dtype=torch.bfloat16) if autocast == "bf16" else nullcontext()
    )
    with torch.inference_mode(), context:
        for index, batch in enumerate(batches, start=1):
            items = [dataset[position] for position in batch]
            mel, lengths, rows = collate(items)
            log_probs, out_lengths = forward(
                model,
                mel.to(device),
                lengths.to(device),
                checkpointing=False,
                training=False,
            )
            texts = _decode_batch(log_probs, out_lengths, lambda ids: decode(tokenizer, ids))
            for row, text in zip(rows, texts, strict=True):
                hyps[row["id"]] = text
            if index % 50 == 0:
                print(f"  {name}: {index}/{len(batches)} batches", flush=True)
    result = score_rows(dataset.rows, hyps, bootstrap=0)
    speakers = {
        speaker: {
            "n": stats["n"],
            "wer": stats["wer"],
            "eta": stats["eta"],
        }
        for speaker, stats in (result["breakdown"].get("speaker") or {}).items()
    }
    return {
        "n": result["overall"]["n"],
        "wer": result["overall"]["wer"],
        "eta": result["overall"]["eta"],
        "per_speaker": speakers,
    }
