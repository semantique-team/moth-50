import architecture_budget as budget
import pytest


def test_parameter_counts_match_the_moth_model():
    # Exact MothSTT counts: 0.8.1 shipped 11,575,169; base+6 measured by probe_export.
    assert budget.acoustic_params(budget.ANCHOR) == 11_575_169
    base6 = budget.Candidate("base +6", 512, 6, 2048, extra=6)
    assert budget.acoustic_params(base6) == 39_259_649


def test_base6_fits_and_base7_now_also_fits_with_the_packed_lm():
    base6 = budget.estimate(budget.Candidate("base +6", 512, 6, 2048, extra=6))
    base7 = budget.estimate(budget.Candidate("base +7", 512, 6, 2048, extra=7))
    assert base6["fits"] == 1.0
    assert base6["headroom_mb"] == pytest.approx(5.32, abs=0.02)
    assert base6["max_lm_mb"] == pytest.approx(8.68, abs=0.02)
    assert base7["fits"] == 1.0
    assert base7["headroom_mb"] == pytest.approx(2.12, abs=0.02)


def test_unmeasured_width_uses_the_conservative_ratio():
    ratio, measured = budget.bytes_per_param(768)
    assert ratio == budget.FALLBACK_BYTES_PER_PARAM
    assert not measured


def test_anchor_reproduces_the_e028_epoch():
    anchor = budget.estimate(budget.ANCHOR)
    assert anchor["flops_vs_moth"] == pytest.approx(1.0)
    # E-028 trained one epoch in 1 h 47; the 10 % pessimism factor gives ~2.0 h.
    assert 1.78 < anchor["hours_per_epoch"] < 2.1


def test_cli_prints_the_table(capsys):
    budget.main(["--all"])
    out = capsys.readouterr().out
    assert "base +6" in out
    assert "50,000,000" in out
