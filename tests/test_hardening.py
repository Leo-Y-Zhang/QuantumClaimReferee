"""Tests for the honesty/default-deny guardrails added after code review.

These exist because the package's whole pitch is "never a silent pass": an invalid
input must raise, never yield a false CERTIFIED, and never silently drop data.
"""

import pytest
from scipy.stats import binom

from qcref.chsh import CLASSICAL_WIN, chsh
from qcref.counts import CountsDataset
from qcref.intervals import clopper_pearson_interval, wilson_interval
from qcref.multiple import holm
from qcref.verdict import Hypothesis, Study


def test_negative_pvalue_cannot_produce_false_certified():
    # The exact hole the review found: correction='none' bypasses array validation,
    # so a negative p would pass `adj_p <= alpha` and falsely CERTIFY.
    with pytest.raises(ValueError):
        Study(correction="none").add_hypothesis(Hypothesis("x", -0.1))
    with pytest.raises(ValueError):
        Study().add("x", 1.5)


def test_summary_shows_na_not_nan_for_unmet():
    v = Study().add("ok", 0.001).add("bad", 0.5, assumptions_met=False).run()
    text = v.summary()
    assert "n/a" in text
    assert "nan" not in text


@pytest.mark.parametrize("level", [1.0, 0.0, 1.5, -0.1])
def test_interval_rejects_out_of_range_level(level):
    with pytest.raises(ValueError):
        wilson_interval(5, 10, level=level)
    with pytest.raises(ValueError):
        clopper_pearson_interval(5, 10, level=level)


def test_all_zero_histogram_rejected():
    with pytest.raises(ValueError):
        CountsDataset({"s": {"00": 0, "01": 0}})


def test_from_qiskit_empty_list_raises():
    class Empty:
        def get_counts(self):
            return []

    with pytest.raises(ValueError):
        CountsDataset.from_qiskit(Empty(), povm="ideal_projective")


def test_from_qiskit_multiple_experiments_warns_and_uses_first():
    class Multi:
        def get_counts(self):
            return [{"0": 5, "1": 5}, {"0": 1, "1": 9}]

    with pytest.warns(UserWarning):
        ds = CountsDataset.from_qiskit(Multi(), povm="ideal_projective")
    assert ds.shots("default") == 10  # only the first experiment


def test_multiple_scalar_raises_valueerror():
    with pytest.raises(ValueError):
        holm(0.05)


def test_shipped_chsh_matches_canonical_formula():
    # Guard against drift between the shipped scalar code and the textbook formula
    # (and, by construction, the vectorized self-test which uses the same binom.sf).
    for wins, rounds in [(66, 80), (6400, 8000), (30, 40)]:
        r = chsh(wins, rounds, setting_randomness_declared=True)
        assert r.p_memory_robust == pytest.approx(
            float(binom.sf(wins - 1, rounds, CLASSICAL_WIN))
        )
