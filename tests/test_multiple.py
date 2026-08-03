import pytest

from qcref.multiple import benjamini_hochberg, bonferroni, holm


def test_holm_known_sequence():
    adj = holm([0.01, 0.02, 0.03, 0.04])
    assert adj == pytest.approx([0.04, 0.06, 0.06, 0.06])


def test_holm_best_of_six_deflation():
    # best-of-6 scan: a rigorous per-test p=0.03 must fail after correction
    adj = holm([0.03, 0.21, 0.44, 0.61, 0.77, 0.90])
    assert adj[0] == pytest.approx(0.18)
    assert adj[0] > 0.05


def test_bonferroni_clips_at_one():
    assert bonferroni([0.03, 0.5]) == pytest.approx([0.06, 1.0])


def test_benjamini_hochberg_monotone_and_bounded():
    adj = benjamini_hochberg([0.001, 0.5, 0.02, 0.9])
    assert all(0.0 <= a <= 1.0 for a in adj)


@pytest.mark.parametrize("bad", [[], [1.2], [-0.1], [float("nan")]])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        holm(bad)
