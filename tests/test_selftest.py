import math

import numpy as np
import pytest
from scipy.stats import norm

from minos.adversary import GreedyDenominatorAdversary, default_adversaries
from minos.power import certification_power
from minos.selftest import (
    AdversarialReport,
    binomial_interval_coverage,
    chsh_adversarial_false_positive_rates,
    chsh_null_false_positive_rates,
    naive_persetting_pvalues,
)


def test_naive_chsh_is_miscalibrated_at_small_n():
    # The thesis of the whole package, checked empirically.
    fpr = chsh_null_false_positive_rates(80, trials=60_000, seed=0)
    assert fpr["memory_robust"] <= 0.055          # rigorous test is valid
    assert fpr["naive_observed"] > 0.055          # naive test over-certifies
    assert fpr["naive_observed"] > fpr["memory_robust"]


def test_naive_and_rigorous_agree_at_large_n():
    fpr = chsh_null_false_positive_rates(8000, trials=60_000, seed=0)
    assert abs(fpr["naive_observed"] - fpr["memory_robust"]) < 0.02


def test_clopper_pearson_is_conservative():
    cov = binomial_interval_coverage("clopper-pearson", 0.5, 100, trials=20_000, seed=1)
    assert cov >= 0.95


def test_wilson_coverage_near_nominal():
    cov = binomial_interval_coverage("wilson", 0.3, 200, trials=20_000, seed=2)
    assert cov >= 0.93


# ------------------------------------------------ naive per-setting analysis


def test_naive_persetting_pvalue_hand_computed():
    # One trial, 20 rounds per setting, win rates (0.9, 0.85, 0.8, 0.75):
    # S_hat = 2 * sum(omega) - 4 = 2.6, SE^2 = sum 4*omega*(1-omega)/20.
    counts = np.array([[20, 20, 20, 20]])
    wins = np.array([[18, 17, 16, 15]])
    omega = wins / counts
    s_hat = 2.0 * omega.sum() - 4.0
    se = math.sqrt(float((4.0 * omega * (1.0 - omega) / counts).sum()))
    expected = float(norm.sf((s_hat - 2.0) / se))
    p = naive_persetting_pvalues(counts, wins)
    assert p.shape == (1,)
    assert p[0] == pytest.approx(expected, rel=1e-12)


def test_naive_persetting_pvalue_never_certifies_an_unseen_setting():
    # A setting pair with zero rounds leaves S_hat undefined; the analysis must
    # return p = 1 (refuse to certify), not divide by zero.
    counts = np.array([[10, 10, 10, 0]])
    wins = np.array([[10, 10, 10, 0]])
    p = naive_persetting_pvalues(counts, wins)
    assert p[0] == 1.0


def test_naive_persetting_pvalue_degenerate_zero_se():
    # All four win rates pinned at 0 or 1 gives SE = 0. S_hat > 2 must map to
    # p = 0 and S_hat <= 2 to p = 1 (sign-aware, no NaN).
    counts = np.array([[10, 10, 10, 10], [10, 10, 10, 10]])
    wins = np.array([[10, 10, 10, 10], [10, 10, 10, 0]])
    p = naive_persetting_pvalues(counts, wins)
    assert p[0] == 0.0  # S_hat = 4
    assert p[1] == 1.0  # S_hat = 2 exactly: not above the local bound


def test_naive_persetting_pvalues_input_validation():
    with pytest.raises(ValueError):
        naive_persetting_pvalues(np.zeros((3, 5)), np.zeros((3, 5)))
    with pytest.raises(ValueError):
        naive_persetting_pvalues(np.zeros((3, 4)), np.zeros((2, 4)))


def test_naive_persetting_pvalues_reject_impossible_tallies():
    # 15 wins of 10 rounds gives negative variance and a NaN SE, which the
    # degenerate branch used to map to a CONFIDENT p = 0 (a certification from
    # unphysical data). Impossible tallies must raise, never certify.
    counts = np.array([[10, 10, 10, 10]])
    with pytest.raises(ValueError):
        naive_persetting_pvalues(counts, np.array([[15, 10, 10, 10]]))  # wins > counts
    with pytest.raises(ValueError):
        naive_persetting_pvalues(counts, np.array([[-5, 10, 10, 10]]))  # negative wins
    with pytest.raises(ValueError):
        naive_persetting_pvalues(np.array([[np.nan, 10, 10, 10]]), np.array([[5, 10, 10, 10]]))


# ------------------------------------------- adversarial memory-loophole mode


def test_adversarial_reports_are_bounded_and_complete():
    n, trials, alpha = 60, 1500, 0.05
    reports = chsh_adversarial_false_positive_rates(n, trials=trials, seed=2)
    assert set(reports) == {a.name for a in default_adversaries()}
    ceiling = certification_power(n, 0.75, alpha=alpha)
    tol = 4.0 * math.sqrt(ceiling * (1.0 - ceiling) / trials)
    for name, rep in reports.items():
        assert isinstance(rep, AdversarialReport)
        assert rep.adversary == name
        assert rep.rounds == n and rep.trials == trials and rep.alpha == alpha
        assert rep.fpr_ceiling_exact == pytest.approx(ceiling)
        # the analyzer bounds every adversary at the exact binomial ceiling
        assert rep.fraction_certified <= ceiling + tol
        # the three verdict fractions partition the runs
        total = rep.fraction_certified + rep.fraction_underpowered + rep.fraction_not_certified
        assert total == pytest.approx(1.0)
        assert 0.0 <= rep.fpr_naive_persetting <= 1.0
        assert 0.0 < rep.mean_win_rate < 1.0


def test_greedy_denominator_blows_up_the_naive_persetting_test():
    # The centrepiece measurement. The greedy-denominator adversary steers its
    # losses into the settings with the largest current counts; that inflates the
    # naive per-setting S_hat AND shrinks its plug-in SE at once. Measured at
    # seed 0: naive FPR 0.2953 at n=80 against a nominal 0.05, while the
    # memory-robust certification rate stays at the exact binomial ceiling.
    reports = chsh_adversarial_false_positive_rates(80, trials=4000, seed=0)
    greedy = reports["greedy_denominator"]
    assert greedy.fpr_naive_persetting > 0.15
    assert greedy.fraction_certified <= greedy.fpr_ceiling_exact + 4.0 * math.sqrt(
        greedy.fpr_ceiling_exact * (1.0 - greedy.fpr_ceiling_exact) / greedy.trials
    )
    # The memoryless saturator is deterministic given the settings: S_hat = 2
    # exactly and SE = 0, so the naive test never fires on it. The danger is
    # specifically HISTORY-DEPENDENT play, which is the loophole's whole point.
    assert reports["memoryless_saturator"].fpr_naive_persetting == 0.0


def test_greedy_denominator_naive_inflation_does_not_vanish_with_n():
    # Unlike the pooled naive-SE gap (which closes by n=8000, see
    # test_naive_and_rigorous_agree_at_large_n), the memory attack on the
    # per-setting estimator scales with its SE: measured naive FPR 0.2953 at
    # n=80 and 0.2873 at n=320 (seed 0). More data does not fix it; only the
    # memory-robust game tail does.
    small = chsh_adversarial_false_positive_rates(80, trials=4000, seed=0)
    large = chsh_adversarial_false_positive_rates(320, trials=4000, seed=0)
    assert small["greedy_denominator"].fpr_naive_persetting > 0.15
    assert large["greedy_denominator"].fpr_naive_persetting > 0.15


def test_adversarial_report_summary_mentions_the_key_numbers():
    reports = chsh_adversarial_false_positive_rates(40, trials=400, seed=1)
    rep = reports["greedy_denominator"]
    text = rep.summary()
    assert "greedy_denominator" in text
    assert f"{rep.fraction_certified:.4f}" in text
    assert f"{rep.fpr_ceiling_exact:.4f}" in text
    assert "naive per-setting" in text


def test_adversarial_rejects_bad_inputs():
    with pytest.raises(ValueError):
        chsh_adversarial_false_positive_rates(0)
    with pytest.raises(ValueError):
        chsh_adversarial_false_positive_rates(40, trials=0)


def test_adversarial_rejects_duplicate_adversary_names():
    # Reports are keyed by adversary name; two same-named adversaries would
    # silently overwrite one scorecard. Refuse up front, before any battery runs.
    with pytest.raises(ValueError):
        chsh_adversarial_false_positive_rates(
            20,
            trials=10,
            adversaries=(GreedyDenominatorAdversary(), GreedyDenominatorAdversary()),
        )
