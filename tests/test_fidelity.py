import pytest

from minos.counts import CountsDataset
from minos.fidelity import fidelity_to_basis_state, probability_interval


def _dataset():
    return CountsDataset.from_counts(
        {"000": 940, "111": 40, "001": 20}, povm="ideal_projective"
    )


def test_probability_interval_contains_point():
    iv = probability_interval(_dataset(), "000")
    assert iv.point == pytest.approx(0.94)
    assert iv.contains(0.94)
    assert any(a.startswith("POVM=") for a in iv.assumptions)


def test_fidelity_to_basis_state_equals_probability():
    ds = _dataset()
    f = fidelity_to_basis_state(ds, "000")
    p = probability_interval(ds, "000")
    assert f.point == pytest.approx(p.point)
    assert "target=computational_basis_state" in f.assumptions


def test_default_deny_without_povm():
    ds = CountsDataset.from_counts({"0": 10, "1": 5})  # no POVM
    with pytest.raises(ValueError):
        probability_interval(ds, "0")


def test_unknown_method_and_setting_raise():
    ds = _dataset()
    with pytest.raises(ValueError):
        probability_interval(ds, "000", method="wald")
    with pytest.raises(ValueError):
        probability_interval(ds, "000", setting="missing")
