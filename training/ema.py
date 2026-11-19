"""Exponential moving average of the trained weights (E-011).

Off by default (``train.ema_decay = 0``), so the E-003/E-005 configs are unchanged.
When on, the trainer updates the average after every optimizer step, evaluates the
averaged weights, selects ``ckpt_best`` on them and stores them as ``ema_model``
next to the raw ``model`` weights (resume always continues from the raw ones).
The readout, calibration and export tools choose with ``--weights``: ``auto``
takes the EMA weights when the checkpoint has them.
"""

from __future__ import annotations

import copy

import torch

WEIGHT_CHOICES = ("auto", "model", "ema")


class WeightEMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)

    def factor(self, step: int) -> float:
        """Decay at ``step``: short horizons early, never longer than ``1 / (1 - decay)``."""
        return min(self.decay, (1.0 + step) / (10.0 + step))

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        weight = 1.0 - self.factor(step)
        for averaged, current in zip(self.model.parameters(), model.parameters(), strict=True):
            averaged.lerp_(current.detach(), weight)

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state, strict=True)


def checkpoint_weights(payload: dict, which: str = "auto") -> tuple[dict, str]:
    """``(state_dict, "model" | "ema")`` from a trainer checkpoint."""
    if which not in WEIGHT_CHOICES:
        raise ValueError(f"weights must be one of {WEIGHT_CHOICES}, got {which!r}")
    ema = payload.get("ema_model")
    if which == "auto":
        which = "ema" if ema is not None else "model"
    if which == "ema" and ema is None:
        raise ValueError("checkpoint has no EMA weights (it was trained with train.ema_decay = 0)")
    return (ema if which == "ema" else payload["model"]), which
