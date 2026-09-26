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


# --- non-integer tallies ----------------------------------------------------------
# chsh() compared wins/rounds with <=, which infinity and fractions pass. With
# wins = rounds = inf, omega and S are nan (so the Tsirelson guard, a > comparison,
# is False) while binom.sf(inf - 1, inf, 3/4) is 0.0 -- a CERTIFIED verdict with
# S = nan for data that contain no rounds at all.


@pytest.mark.parametrize(
    "wins, rounds",
    [
        (float("inf"), float("inf")),
        (1e300, float("inf")),
        (69.5, 80),
        (66, 80.5),
        (True, True),
    ],
)
def test_chsh_rejects_non_integer_tallies(wins, rounds):
    with pytest.raises(ValueError, match="integer"):
        chsh(wins, rounds, setting_randomness_declared=True)


def test_chsh_accepts_numpy_integers_and_integral_floats():
    import numpy as np

    ref = chsh(6400, 8000, setting_randomness_declared=True)
    for wins, rounds in [(np.int64(6400), np.int64(8000)), (6400.0, 8000.0)]:
        r = chsh(wins, rounds, setting_randomness_declared=True)
        assert r.certified
        assert r.p_memory_robust == ref.p_memory_robust


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.3, True])
def test_wins_from_setting_counts_rejects_non_integer_counts(bad):
    from qcref.chsh import wins_from_setting_counts

    counts = {(x, y): {(0, 0): 10} for x in (0, 1) for y in (0, 1)}
    counts[(1, 1)] = {(0, 1): bad}
    with pytest.raises(ValueError, match="integer"):
        wins_from_setting_counts(counts)


# --- p-values that are not numbers -------------------------------------------------
# bool is an int subclass, so Study.add("x", False) was a p-value of 0 and
# CERTIFIED: passing a flag such as ``result.certified`` where the p-value belongs
# certified exactly the results that had *failed*.


@pytest.mark.parametrize("flag", [False, True])
def test_study_rejects_boolean_pvalue(flag):
    import numpy as np

    with pytest.raises(ValueError, match="p-value"):
        Study().add("x", flag)
    with pytest.raises(ValueError, match="p-value"):
        Study().add("x", np.bool_(flag))


def test_study_still_accepts_numpy_float_pvalue():
    import numpy as np

    assert Study().add("x", np.float64(0.001)).run().certified


# --- counts that are not shot counts -----------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.5, True])
def test_counts_dataset_rejects_non_integer_counts(bad):
    with pytest.raises(ValueError, match="integer"):
        CountsDataset({"s": {"0": bad, "1": 3}})


def test_counts_dataset_accepts_numpy_integer_counts():
    import numpy as np

    ds = CountsDataset({"s": {"0": np.int64(3), "1": np.int32(7)}})
    assert ds.shots("s") == 10


# --- a target that is not a state of the measured register -------------------------
# fidelity_to_basis_state() looked the target up with hist.get(bitstring, 0), so a
# target with the wrong qubit count (or not a bitstring at all) was scored as
# "measured zero times": a confident interval near fidelity 0 for a state the
# register cannot even be in, instead of an error.


@pytest.mark.parametrize("target", ["0", "000", "2x", "", "0 1"])
def test_fidelity_rejects_target_of_wrong_width_or_alphabet(target):
    from qcref.fidelity import fidelity_to_basis_state, probability_interval

    ds = CountsDataset.from_counts({"00": 90, "11": 10}, povm="ideal_projective")
    with pytest.raises(ValueError, match="bitstring"):
        fidelity_to_basis_state(ds, target)
    with pytest.raises(ValueError, match="bitstring"):
        probability_interval(ds, target)


def test_fidelity_unobserved_but_valid_target_is_still_zero_count():
    from qcref.fidelity import fidelity_to_basis_state

    ds = CountsDataset.from_counts({"00": 90, "11": 10}, povm="ideal_projective")
    iv = fidelity_to_basis_state(ds, "01")
    assert iv.point == 0.0 and iv.n == 100


@pytest.mark.parametrize("k, n", [(0.5, 1), (1, 2.5), (float("inf"), float("inf")), (True, 2)])
def test_intervals_reject_non_integer_k_n(k, n):
    with pytest.raises(ValueError, match="integer"):
        wilson_interval(k, n)
    with pytest.raises(ValueError, match="integer"):
        clopper_pearson_interval(k, n)


def test_from_qiskit_does_not_truncate_fractional_counts():
    # int(2.7) == 2: quasi-probabilities or rescaled counts were silently truncated.
    class Frac:
        def get_counts(self):
            return {"0": 2.7, "1": 7.3}

    with pytest.raises(ValueError, match="integer"):
        CountsDataset.from_qiskit(Frac(), povm="ideal_projective")
