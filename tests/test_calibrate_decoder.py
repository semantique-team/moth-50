import pytest
from calibrate_decoder import grid_settings, parse_values, select_setting
from frozen_readout import FROZEN


def test_grid_always_contains_the_frozen_point():
    settings = grid_settings([0.35, 0.45], [2.0])
    assert len(settings) == 3
    assert settings[-1] == FROZEN
    assert FROZEN in grid_settings([0.55], [2.7])
    assert len(grid_settings([0.55], [2.7])) == 1
    assert {setting.beam_width for setting in settings} == {12}
    assert {setting.top_k for setting in settings} == {10}


def test_parse_values():
    assert parse_values("0.25, 0.5,") == [0.25, 0.5]
    with pytest.raises(ValueError):
        parse_values(" , ")


def _point(lm, bonus, *, macro, macro_ci=None, s1=None, frozen=False):
    return {
        "lm_weight": lm,
        "token_bonus": bonus,
        "frozen": frozen,
        "macro_wer": macro,
        "macro_vs_frozen": None if frozen else {"point": 0.0, "ci": macro_ci},
        "s1": s1,
    }


def _s1(wer, eta, wer_ci, eta_ci=(-0.01, 0.01)):
    return {
        "wer": wer,
        "eta": eta,
        "vs_frozen": {
            "wer_diff": 0.0,
            "wer_diff_ci": list(wer_ci),
            "eta_diff": 0.0,
            "eta_diff_ci": list(eta_ci),
        },
    }


def test_s1_dev_decides_and_dev_macro_guards():
    frozen = _point(0.55, 2.7, macro=0.176, frozen=True, s1=_s1(0.20, 0.60, (0, 0)))
    better = _point(
        0.45, 2.0, macro=0.175, macro_ci=[-0.003, 0.001], s1=_s1(0.18, 0.62, (-0.03, -0.01))
    )
    best_s1_but_macro_worse = _point(
        0.25, 1.5, macro=0.190, macro_ci=[0.008, 0.020], s1=_s1(0.17, 0.63, (-0.04, -0.01))
    )
    selection = select_setting([frozen, better, best_s1_but_macro_worse])
    assert selection["objective"] == "s1-dev WER"
    assert selection["eligible"] == 2
    assert selection["best"] == {"lm_weight": 0.45, "token_bonus": 2.0}
    assert selection["adopt"] is True
    assert selection["selected"] == {"lm_weight": 0.45, "token_bonus": 2.0}


def test_s1_eta_regression_is_excluded():
    frozen = _point(0.55, 2.7, macro=0.176, frozen=True, s1=_s1(0.20, 0.60, (0, 0)))
    eta_worse = _point(
        0.25,
        3.5,
        macro=0.176,
        macro_ci=[-0.001, 0.001],
        s1=_s1(0.18, 0.50, (-0.03, -0.01), eta_ci=(-0.15, -0.05)),
    )
    selection = select_setting([frozen, eta_worse])
    assert selection["eligible"] == 1
    assert selection["adopt"] is False
    assert selection["selected"] == {"lm_weight": 0.55, "token_bonus": 2.7}


def test_a_tie_keeps_the_frozen_settings():
    frozen = _point(0.55, 2.7, macro=0.176, frozen=True, s1=_s1(0.20, 0.60, (0, 0)))
    tie = _point(
        0.45, 2.7, macro=0.176, macro_ci=[-0.002, 0.002], s1=_s1(0.199, 0.60, (-0.01, 0.01))
    )
    selection = select_setting([frozen, tie])
    assert selection["best"] == {"lm_weight": 0.45, "token_bonus": 2.7}
    assert selection["adopt"] is False
    assert selection["selected"] == {"lm_weight": 0.55, "token_bonus": 2.7}


def test_without_s1_dev_macro_decides():
    frozen = _point(0.55, 2.7, macro=0.176, frozen=True)
    better = _point(0.35, 2.0, macro=0.170, macro_ci=[-0.008, -0.004])
    selection = select_setting([frozen, better])
    assert selection["objective"] == "dev-macro WER"
    assert selection["adopt"] is True
    assert selection["selected"] == {"lm_weight": 0.35, "token_bonus": 2.0}
