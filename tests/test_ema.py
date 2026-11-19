import pytest

pytest.importorskip("torch")

import torch

from training.ema import WeightEMA, checkpoint_weights


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3))


def test_ema_starts_as_a_frozen_copy():
    model = _model()
    ema = WeightEMA(model, 0.999)
    for averaged, current in zip(ema.model.parameters(), model.parameters(), strict=True):
        assert torch.equal(averaged, current)
        assert averaged.data_ptr() != current.data_ptr()
        assert averaged.requires_grad is False
    assert ema.model.training is False


def test_ema_update_is_a_lerp_with_a_warmup():
    model = _model()
    ema = WeightEMA(model, 0.99)
    assert ema.factor(0) == pytest.approx(0.1)
    assert ema.factor(10_000) == pytest.approx(0.99)
    before = [parameter.clone() for parameter in ema.model.parameters()]
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.update(model, step=10_000)
    for start, averaged, current in zip(before, ema.model.parameters(), model.parameters()):
        assert torch.allclose(averaged, 0.99 * start + 0.01 * current)


def test_ema_state_round_trip():
    model = _model()
    ema = WeightEMA(model, 0.9)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(3.0)
    ema.update(model, step=100)
    other = WeightEMA(_model(), 0.9)
    other.load_state_dict(ema.state_dict())
    for left, right in zip(other.model.parameters(), ema.model.parameters(), strict=True):
        assert torch.equal(left, right)


def test_decay_must_be_inside_the_unit_interval():
    for decay in (0.0, 1.0, -0.5):
        with pytest.raises(ValueError, match="ema_decay"):
            WeightEMA(_model(), decay)


def test_checkpoint_weights_selection():
    raw, averaged = {"w": 1}, {"w": 2}
    assert checkpoint_weights({"model": raw}) == (raw, "model")
    assert checkpoint_weights({"model": raw, "ema_model": None}) == (raw, "model")
    with_ema = {"model": raw, "ema_model": averaged}
    assert checkpoint_weights(with_ema) == (averaged, "ema")
    assert checkpoint_weights(with_ema, "model") == (raw, "model")
    with pytest.raises(ValueError, match="no EMA"):
        checkpoint_weights({"model": raw}, "ema")
    with pytest.raises(ValueError, match="weights"):
        checkpoint_weights(with_ema, "best")
