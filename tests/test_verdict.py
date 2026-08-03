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
