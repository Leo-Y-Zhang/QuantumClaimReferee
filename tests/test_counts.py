import pytest

from minos.counts import CountsDataset


def test_from_counts_and_probability():
    ds = CountsDataset.from_counts({"00": 480, "11": 470, "01": 50}, povm="ideal_projective")
    assert ds.shots("default") == 1000
    assert ds.probability("default", "00") == pytest.approx(0.48)
    assert ds.probability("default", "10") == 0.0  # unseen outcome


def test_require_povm_default_deny():
    ds = CountsDataset.from_counts({"0": 10, "1": 5})
    with pytest.raises(ValueError):
        ds.require_povm()
    ds2 = CountsDataset.from_counts({"0": 10, "1": 5}, povm="ideal_projective")
    assert ds2.require_povm() == "ideal_projective"


def test_rejects_inconsistent_bitstrings():
    with pytest.raises(ValueError):
        CountsDataset({"s": {"00": 5, "1": 3}})
    with pytest.raises(ValueError):
        CountsDataset({"s": {"0x": 5}})


def test_from_qiskit_ducktype():
    class FakeResult:
        def get_counts(self):
            return {"0 0": 10, "1 1": 6}  # spaced multi-register style

    ds = CountsDataset.from_qiskit(FakeResult(), povm="ideal_projective")
    assert ds.shots("default") == 16
    assert ds.probability("default", "00") == pytest.approx(10 / 16)
