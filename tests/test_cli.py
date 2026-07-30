import pytest

from minos.cli import main


def test_chsh_certified_exit_zero(capsys):
    rc = main(["chsh", "--wins", "6400", "--rounds", "8000", "--randomised"])
    assert rc == 0
    assert "CERTIFIED" in capsys.readouterr().out


def test_selftest_bad_n_is_usage_error():
    with pytest.raises(SystemExit) as exc:
        main(["selftest", "--n", "0"])
    assert exc.value.code == 2  # argparse usage error, not a traceback


def test_chsh_underpowered_exit_one():
    rc = main(["chsh", "--wins", "66", "--rounds", "80", "--randomised"])
    assert rc == 1


def test_chsh_default_deny_without_randomised():
    rc = main(["chsh", "--wins", "6400", "--rounds", "8000"])
    assert rc == 1  # ASSUMPTIONS_UNMET is not certified


def test_selftest_runs(capsys):
    rc = main(["selftest", "--n", "80", "--trials", "3000"])
    assert rc == 0
    assert "false-positive" in capsys.readouterr().out


def test_chsh_underpowered_summary_includes_plan_hint(capsys):
    rc = main(["chsh", "--wins", "66", "--rounds", "80", "--randomised"])
    assert rc == 1
    assert "90% power" in capsys.readouterr().out


def test_plan_with_s(capsys):
    rc = main(["plan", "--S", "2.4", "--alpha", "0.05", "--power", "0.9"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "604" in out  # rounds needed
    assert "471" in out  # implied critical win count


def test_plan_with_win_rate(capsys):
    rc = main(["plan", "--win-rate", "0.9"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PLAN: 55 rounds" in out


def test_plan_requires_exactly_one_of_s_and_win_rate():
    with pytest.raises(SystemExit) as exc:
        main(["plan", "--S", "2.4", "--win-rate", "0.9"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(["plan"])
    assert exc.value.code == 2


def test_plan_rejects_s_at_or_below_local_bound():
    with pytest.raises(SystemExit) as exc:
        main(["plan", "--S", "2.0"])
    assert exc.value.code == 2


def test_demo_runs():
    assert main(["demo"]) == 0


def test_requires_subcommand():
    with pytest.raises(SystemExit):
        main([])
