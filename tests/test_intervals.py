import pytest

from qreferee.intervals import clopper_pearson_interval, wilson_interval


def test_wilson_known_values():
    iv = wilson_interval(8, 10, level=0.95)
    assert iv.point == pytest.approx(0.8)
    assert iv.lo == pytest.approx(0.490, abs=2e-3)
    assert iv.hi == pytest.approx(0.943, abs=2e-3)
    assert iv.contains(0.8)


def test_wilson_stays_in_unit_interval_near_edges():
    iv = wilson_interval(10, 10)
    assert 0.0 <= iv.lo <= iv.hi <= 1.0
    assert iv.hi < 1.0  # Wilson does not collapse to a zero-width point at k==n


def test_clopper_pearson_edges():
    lo0 = clopper_pearson_interval(0, 20)
    assert lo0.lo == 0.0 and 0.0 < lo0.hi < 1.0
    hin = clopper_pearson_interval(20, 20)
    assert hin.hi == 1.0 and 0.0 < hin.lo < 1.0


def test_clopper_pearson_wider_than_wilson():
    w = wilson_interval(5, 20)
    cp = clopper_pearson_interval(5, 20)
    assert (cp.hi - cp.lo) >= (w.hi - w.lo)


@pytest.mark.parametrize("k,n", [(-1, 10), (11, 10), (0, 0)])
def test_invalid_inputs_raise(k, n):
    with pytest.raises(ValueError):
        wilson_interval(k, n)
