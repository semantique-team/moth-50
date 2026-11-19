import dataclasses

import pytest
from frozen_readout import (
    FROZEN,
    DecoderSettings,
    gate_verdict,
    macro_wer,
    no_regression_verdict,
    read_hyps,
    resolve_decoder_settings,
    write_hyps,
)


def test_macro_wer_is_the_mean_of_it_and_en():
    assert macro_wer({"it": 0.20, "en": 0.10}) == pytest.approx(0.15)
    with pytest.raises(ValueError, match="it/en"):
        macro_wer({"it": 0.20})


def test_gate_needs_1pp_and_a_ci_excluding_zero():
    passed = gate_verdict(-0.015, [-0.020, -0.010])
    assert passed["pass"] is True
    assert passed["improvement_pp"] == 1.5
    assert gate_verdict(-0.015, [-0.020, 0.001])["pass"] is False
    assert gate_verdict(-0.005, [-0.010, -0.001])["pass"] is False
    assert gate_verdict(-0.020, None)["pass"] is False


def test_frozen_decoder_is_the_e002_protocol():
    assert dataclasses.asdict(FROZEN) == {
        "beam_width": 12,
        "top_k": 10,
        "lm_weight": 0.55,
        "token_bonus": 2.7,
    }


def _paired(wer_diff, wer_ci, eta_diff=None, eta_ci=None):
    return {
        "wer_diff": wer_diff,
        "wer_diff_ci": wer_ci,
        "eta_diff": eta_diff,
        "eta_diff_ci": eta_ci,
    }


def test_s1_dev_regression_needs_a_ci_excluding_zero():
    better = no_regression_verdict(_paired(-0.03, [-0.05, -0.01], 0.04, [0.01, 0.07]))
    assert better["pass"] is True
    assert better["wer_diff_pp"] == -3.0
    tie = no_regression_verdict(_paired(0.01, [-0.01, 0.03], -0.01, [-0.03, 0.01]))
    assert tie["pass"] is True
    assert tie["wer_regression"] is False and tie["eta_regression"] is False


def test_s1_dev_wer_or_eta_regression_fails():
    worse_wer = no_regression_verdict(_paired(0.02, [0.005, 0.035], 0.0, [-0.01, 0.01]))
    assert worse_wer["wer_regression"] is True
    assert worse_wer["pass"] is False
    worse_eta = no_regression_verdict(_paired(-0.02, [-0.04, -0.01], -0.05, [-0.08, -0.02]))
    assert worse_eta["eta_regression"] is True
    assert worse_eta["pass"] is False


def test_no_ci_means_no_verdict():
    verdict = no_regression_verdict(_paired(0.01, None))
    assert verdict["has_ci"] is False
    assert verdict["pass"] is None


def test_hypotheses_round_trip(tmp_path):
    hyps = {"a": "ho spostato il meeting", "b": ""}
    path = tmp_path / "out" / "hyp.jsonl"
    write_hyps(path, hyps)
    assert read_hyps(path) == hyps


def test_default_decoder_is_frozen():
    assert resolve_decoder_settings() == FROZEN


def test_experiment_decoder_needs_an_explicit_opt_in_and_both_values():
    with pytest.raises(ValueError, match="experiment"):
        resolve_decoder_settings(lm_weight=0.45, token_bonus=2.0)
    with pytest.raises(ValueError, match="both"):
        resolve_decoder_settings(lm_weight=0.45, experiment=True)
    with pytest.raises(ValueError, match="non-negative"):
        resolve_decoder_settings(lm_weight=-0.1, token_bonus=2.0, experiment=True)


def test_experiment_decoder_keeps_beam_and_top_k():
    settings = resolve_decoder_settings(lm_weight=0.45, token_bonus=2.0, experiment=True)
    assert settings == DecoderSettings(beam_width=12, top_k=10, lm_weight=0.45, token_bonus=2.0)
