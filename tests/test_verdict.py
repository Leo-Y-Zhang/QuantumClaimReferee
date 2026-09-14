import pytest

from qcref.verdict import Study


def test_all_pass_is_certified():
    v = Study(alpha=0.05, correction="holm").add("a", 0.001).add("b", 0.002).run()
    assert v.classification == "CERTIFIED"
    assert v.certified


def test_one_fail_is_not_certified():
    v = Study().add("a", 0.001).add("b", 0.40).run()
    assert v.classification == "NOT_CERTIFIED"


def test_unmet_assumptions_dominate():
    v = (
        Study()
        .add("a", 0.001)
        .add("b", 0.001, assumptions_met=False, detail="POVM undeclared")
        .run()
    )
    assert v.classification == "ASSUMPTIONS_UNMET"
    unmet = [r for r in v.results if r.name == "b"][0]
    assert unmet.status == "ASSUMPTIONS_UNMET"


def test_correction_can_flip_a_verdict():
    # six borderline hypotheses: individually one passes, but Holm deflates it
    study = Study(alpha=0.05, correction="holm")
    for i, p in enumerate([0.03, 0.21, 0.44, 0.61, 0.77, 0.90]):
        study.add(f"pair_{i}", p)
    v = study.run()
    assert v.classification == "NOT_CERTIFIED"
    best = v.results[0]
    assert best.raw_p == pytest.approx(0.03)
    assert best.adjusted_p == pytest.approx(0.18)


def test_empty_study_raises():
    with pytest.raises(ValueError):
        Study().run()


def test_benjamini_hochberg_correction_end_to_end_not_certified():
    # Same family used to hand-verify benjamini_hochberg() in test_multiple.py:
    # p = [0.001, 0.045, 0.2, 0.6]. The correct BH adjustment (m=4) gives
    # [0.004, 0.09, 0.266666..., 0.6] -- only the first hypothesis clears
    # alpha=0.05, so the study-wide, default-deny verdict must be
    # NOT_CERTIFIED (CERTIFIED requires *every* hypothesis to clear the bar).
    # A broken benjamini_hochberg() that deflates every p-value by roughly an
    # extra factor of m (dividing instead of multiplying) pushes all four
    # adjusted values under alpha and would wrongly report CERTIFIED here.
    study = Study(alpha=0.05, correction="benjamini-hochberg")
    for name, p in [("a", 0.001), ("b", 0.045), ("c", 0.2), ("d", 0.6)]:
        study.add(name, p)
    v = study.run()
    assert v.classification == "NOT_CERTIFIED"
    by_name = {r.name: r for r in v.results}
    assert by_name["a"].status == "CERTIFIED"
    assert by_name["d"].status == "NOT_CERTIFIED"
    assert by_name["d"].adjusted_p == pytest.approx(0.6)


@pytest.mark.parametrize("alpha", [1.0, 0.0])
def test_alpha_outside_open_unit_interval_raises(alpha):
    with pytest.raises(ValueError):
        Study(alpha=alpha)
