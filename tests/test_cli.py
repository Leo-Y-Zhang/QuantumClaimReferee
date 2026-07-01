import pytest

from qreferee.cli import main


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


def test_demo_runs():
    assert main(["demo"]) == 0


def test_requires_subcommand():
    with pytest.raises(SystemExit):
        main([])
