from qreferee import __version__
from qreferee.report import referee_report
from qreferee.verdict import Study


def _verdict():
    return Study().add("ghz_fidelity", 0.001, estimate="F=0.95").add("chsh", 0.002).run()


def test_report_is_deterministic():
    v = _verdict()
    assert referee_report(v) == referee_report(v)


def test_report_contains_headline_and_versions():
    text = referee_report(_verdict(), title="run-1")
    assert "VERDICT: CERTIFIED" in text
    assert "run-1" in text
    assert __version__ in text
    assert "sha256:" in text
    assert "CAVEATS" in text


def test_report_renders_unmet_as_na():
    v = Study().add("x", 0.001).add("y", 0.5, assumptions_met=False).run()
    text = referee_report(v)
    assert "ASSUMPTIONS_UNMET" in text
    assert "n/a" in text
