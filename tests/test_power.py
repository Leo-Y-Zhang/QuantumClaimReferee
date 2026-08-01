import math

import numpy as np
import pytest

from minos.chsh import chsh, s_to_omega
from minos.power import PlanResult, certification_power, critical_wins, plan_rounds


def _exact_upper_tail(k: int, n: int, p: float) -> float:
    """Independent exact binomial sum P[Bin(n, p) >= k], no scipy."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.comb(n, j) * p**j * (1.0 - p) ** (n - j) for j in range(k, n + 1))


def _independent_critical_wins(n: int, alpha: float) -> int:
    """Smallest k with P[Bin(n, 3/4) >= k] <= alpha, by brute force; n+1 if none."""
    for k in range(0, n + 1):
        if _exact_upper_tail(k, n, 0.75) <= alpha:
            return k
    return n + 1


# ---------------------------------------------------------------- critical_wins


def test_critical_wins_hand_computed_smallest_certifiable_n():
    # P[Bin(10, 3/4) >= 10] = 0.75^10 = 0.0563 > 0.05: even perfect play cannot
    # certify at n=10, so the critical count is the unattainable n+1 = 11.
    assert 0.75**10 > 0.05
    assert critical_wins(10, 0.05) == 11
    # P[Bin(11, 3/4) >= 11] = 0.75^11 = 0.0422 <= 0.05, and
    # P[Bin(11, 3/4) >= 10] = 0.75^11 + 11 * 0.75^10 * 0.25 = 0.197 > 0.05,
    # so at n=11 the critical count is exactly 11 (only a perfect run certifies).
    assert 0.75**11 <= 0.05 < 0.75**11 + 11 * 0.75**10 * 0.25
    assert critical_wins(11, 0.05) == 11


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_critical_wins_matches_independent_binomial_sum(alpha):
    for n in range(1, 61):
        assert critical_wins(n, alpha) == _independent_critical_wins(n, alpha)


@pytest.mark.parametrize("n", [5, 11, 40, 100])
def test_critical_wins_agrees_with_shipped_verdict(n):
    # The acceptance region is now the conjunction of two conditions, and this
    # pins BOTH: the p-value threshold, and physical plausibility. chsh refuses a
    # win count implying S above the Tsirelson bound however small its p-value,
    # because no quantum device can produce one -- so critical_wins alone stopped
    # being the whole story. It answers an arithmetic question and cannot see
    # physics; that is why it is not the sole gate.
    from minos.chsh import TSIRELSON_S, omega_to_s

    c = critical_wins(n, 0.05)
    for wins in range(n + 1):
        r = chsh(wins, n, alpha=0.05, setting_randomness_declared=True)
        physical = omega_to_s(wins / n) <= TSIRELSON_S
        assert r.certified == (wins >= c and physical)


@pytest.mark.parametrize("kwargs", [{"rounds": 0}, {"rounds": -3}])
def test_critical_wins_rejects_nonpositive_rounds(kwargs):
    with pytest.raises(ValueError):
        critical_wins(alpha=0.05, **kwargs)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, float("nan")])
def test_critical_wins_rejects_bad_alpha(alpha):
    with pytest.raises(ValueError):
        critical_wins(100, alpha)


# ---------------------------------------------------------- certification_power


def test_certification_power_hand_computed():
    # At n=11, alpha=0.05 the critical count is 11 (see above), so the power at
    # win rate p is exactly P[Bin(11, p) = 11] = p^11.
    assert certification_power(11, 0.9, alpha=0.05) == pytest.approx(0.9**11)
    assert certification_power(11, 1.0, alpha=0.05) == pytest.approx(1.0)
    # Below the smallest certifiable n the power is exactly zero.
    assert certification_power(10, 1.0, alpha=0.05) == 0.0


def test_certification_power_matches_independent_binomial_sum():
    for n in (11, 25, 55, 80):
        c = _independent_critical_wins(n, 0.05)
        for p in (0.8, 0.85, 0.9):
            assert certification_power(n, p, alpha=0.05) == pytest.approx(
                _exact_upper_tail(c, n, p), abs=1e-12
            )


# ------------------------------------------------------------------ plan_rounds


def test_plan_rounds_perfect_win_rate_hand_computed():
    # The arithmetic answer is 11 -- the smallest n with 0.75^n <= alpha -- and it
    # is not a usable plan: certifying at n=11 needs 11 wins from 11, i.e. S=4.0,
    # the algebraic PR-box maximum, which no quantum device can reach. A plan is
    # advice, and advice that cannot be followed is not advice. The planner now
    # skips every n whose threshold implies S above the Tsirelson bound, so the
    # answer is the floor: 60 rounds, certifying at 51 wins (S = 2.800).
    plan = plan_rounds(1.0, alpha=0.05, power=0.9)
    assert isinstance(plan, PlanResult)
    assert plan.rounds == 60
    assert plan.critical_wins == 51
    assert plan.power == pytest.approx(1.0)


def test_plan_rounds_is_minimal_against_independent_computation():
    # Verify minimality exhaustively with the independent no-scipy computation.
    # Minimal now means minimal among the PHYSICALLY ATTAINABLE n, so smaller n
    # are excluded by the Tsirelson condition rather than by insufficient power.
    from minos.chsh import TSIRELSON_S, omega_to_s

    plan = plan_rounds(0.9, alpha=0.05, power=0.9)
    for n in range(1, plan.rounds):
        c = _independent_critical_wins(n, 0.05)
        unreachable = c > n or omega_to_s(c / n) > TSIRELSON_S
        assert unreachable or _exact_upper_tail(c, n, 0.9) < 0.9
    c = _independent_critical_wins(plan.rounds, 0.05)
    assert _exact_upper_tail(c, plan.rounds, 0.9) >= 0.9


def test_plan_rounds_sawtooth_regression():
    # Exact binomial power is NON-monotone in n (sawtooth): whenever the critical
    # count steps up by one, the power momentarily drops. Documented case at
    # p=0.9, alpha=0.05, target 0.9:
    #   power(54) = 0.8321   (c=47)
    #   power(55) = 0.9056   (c=47)  <- first crossing, the minimal n
    #   power(56) = 0.8970   (c=48)  <- dips back BELOW the target
    # A bisection assuming monotone power can land on 56, see failure, and search
    # upward past the true answer. The scan must return 55.
    assert certification_power(54, 0.9, alpha=0.05) == pytest.approx(0.8321, abs=5e-4)
    assert certification_power(55, 0.9, alpha=0.05) == pytest.approx(0.9056, abs=5e-4)
    assert certification_power(56, 0.9, alpha=0.05) == pytest.approx(0.8970, abs=5e-4)
    assert certification_power(56, 0.9, alpha=0.05) < 0.9
    # The power figures above are unchanged -- the sawtooth is a property of the
    # binomial, not of the gate. What changed is which n the scan may return: 55
    # is below the physical floor of 60 at this alpha, so the plan is 60/51.
    plan = plan_rounds(0.9, alpha=0.05, power=0.9)
    assert plan.rounds == 60
    assert plan.critical_wins == 51


def test_plan_rounds_from_s_value():
    # S=2.4 maps to omega=0.8; documented answer n=604, c=471, power ~0.9007.
    plan = plan_rounds(s_to_omega(2.4), alpha=0.05, power=0.9)
    assert plan.rounds == 604
    assert plan.critical_wins == 471
    assert plan.power == pytest.approx(0.9007, abs=5e-4)
    assert plan.S == pytest.approx(2.4)


def test_plan_rounds_monte_carlo_certification_rate():
    # In the style of minos.selftest: simulate at the hypothesised win rate over
    # the planned n and check the empirical certification rate clears the target
    # within Monte-Carlo tolerance (3 sigma of the binomial MC error).
    trials = 20_000
    plan = plan_rounds(0.85, alpha=0.05, power=0.9)
    rng = np.random.default_rng(0)
    wins = rng.binomial(plan.rounds, 0.85, size=trials)
    rate = float(np.mean(wins >= plan.critical_wins))
    tol = 3.0 * math.sqrt(plan.power * (1.0 - plan.power) / trials)
    assert rate >= 0.9 - tol
    # and the empirical rate matches the exact power, not just the target
    assert rate == pytest.approx(plan.power, abs=tol)


def test_plan_rounds_rejects_win_rate_at_or_below_classical_bound():
    # At or below omega = 3/4 the certification is calibrated: the certify
    # probability never exceeds alpha, so no n can reach a power target > alpha.
    for p in (0.75, 0.5, 0.0):
        with pytest.raises(ValueError):
            plan_rounds(p, alpha=0.05, power=0.9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"win_rate": 1.1},
        {"win_rate": 0.9, "alpha": 0.0},
        {"win_rate": 0.9, "alpha": 1.0},
        {"win_rate": 0.9, "power": 0.0},
        {"win_rate": 0.9, "power": 1.0},
        {"win_rate": 0.9, "max_rounds": 0},
    ],
)
def test_plan_rounds_rejects_out_of_range_inputs(kwargs):
    with pytest.raises(ValueError):
        plan_rounds(**kwargs)


def test_plan_rounds_raises_when_scan_cap_exceeded():
    with pytest.raises(ValueError, match="max_rounds"):
        plan_rounds(0.76, alpha=0.05, power=0.9, max_rounds=50)


def test_plan_result_summary_mentions_the_numbers():
    plan = plan_rounds(0.9, alpha=0.05, power=0.9)
    text = plan.summary()
    assert "60" in text
    assert "51" in text
    assert "0.05" in text
