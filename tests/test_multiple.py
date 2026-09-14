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


def test_benjamini_hochberg_known_values():
    # Hand-computed BH adjustment for p = [0.2, 0.001, 0.6, 0.045], m=4.
    #
    # Sort ascending and assign ranks:
    #   rank 1: p=0.001  (orig idx 1)
    #   rank 2: p=0.045  (orig idx 3)
    #   rank 3: p=0.2    (orig idx 0)
    #   rank 4: p=0.6    (orig idx 2)
    #
    # Scale each by m/rank:
    #   rank 1: 0.001 * 4/1 = 0.004
    #   rank 2: 0.045 * 4/2 = 0.09
    #   rank 3: 0.2   * 4/3 = 0.266666...
    #   rank 4: 0.6   * 4/4 = 0.6
    #
    # Enforce monotonicity with a running minimum from the largest rank down
    # to the smallest (an adjusted p can only shrink, never grow, as you move
    # to a smaller raw p):
    #   rank 4: 0.6
    #   rank 3: min(0.6, 0.266666...)  = 0.266666...
    #   rank 2: min(0.266666..., 0.09) = 0.09
    #   rank 1: min(0.09, 0.004)       = 0.004
    #
    # Mapped back to the original order [0.2, 0.001, 0.6, 0.045]:
    #   idx 0 (rank 3) -> 0.266666...
    #   idx 1 (rank 1) -> 0.004
    #   idx 2 (rank 4) -> 0.6
    #   idx 3 (rank 2) -> 0.09
    adj = benjamini_hochberg([0.2, 0.001, 0.6, 0.045])
    assert adj == pytest.approx([0.8 / 3, 0.004, 0.6, 0.09])
    # The largest (least significant) p-value is at rank m, where scaling by
    # m/rank is a no-op, so its adjusted value must equal its raw value.
    assert adj[2] == pytest.approx(0.6)


@pytest.mark.parametrize("bad", [[], [1.2], [-0.1], [float("nan")]])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        holm(bad)
