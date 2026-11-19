import pytest
from frozen_readout import gate_verdict, macro_wer


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
