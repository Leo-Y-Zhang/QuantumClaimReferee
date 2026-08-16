import math

import numpy as np
import pytest

from qcref.chsh import chsh, s_to_omega
from qcref.power import PlanResult, certification_power, critical_wins, plan_rounds
from qcref.status import CERTIFIED


def _exact_upper_tail(k: int, n: int, p: float) -> float:
    """Independent exact binomial sum P[Bin(n, p) >= k], no scipy."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.comb(n, j) * p**j * (1.0 - p) ** (n - j) for j in range(k, n + 1))


def _independent_critical_wins(n: int, alpha: float) -> int:
    """Smallest k with P[Bin(n, 3/4) >= k] <= alpha, by brute force; n+1 if none.

    Searched downward from the unattainable n+1 (whose tail is 0): the tail is
    decreasing in k, so the qualifying set is upward-closed and this lands on the
    same k an upward scan would, in a few dozen steps instead of n.
    """
    k = n + 1
    while k > 0 and _exact_upper_tail(k - 1, n, 0.75) <= alpha:
        k -= 1
    return k


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
    from qcref.chsh import TSIRELSON_S, omega_to_s

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


def test_plan_rounds_at_the_tsirelson_bound_hand_computed():
    # The strongest hypothesis a quantum device can offer is exactly the Tsirelson
    # bound, and the planner accepts it -- but only just. A device sitting on the
    # bound overshoots it in half of all runs by sampling noise alone, and chsh
    # refuses those as ASSUMPTIONS_UNMET, so the certification rate cannot be
    # bought past ~1/2 however many rounds are spent: 198 rounds buy 0.5098, and
    # 90% is out of reach at any n. A plan is advice, and advice that cannot be
    # followed is not advice.
    from qcref.chsh import TSIRELSON_S

    plan = plan_rounds(s_to_omega(TSIRELSON_S), alpha=0.05, power=0.5, max_rounds=1000)
    assert isinstance(plan, PlanResult)
    assert plan.rounds == 198
    assert plan.critical_wins == 159
    assert plan.power == pytest.approx(0.5098, abs=5e-4)
    with pytest.raises(ValueError, match="max_rounds"):
        plan_rounds(s_to_omega(TSIRELSON_S), alpha=0.05, power=0.9, max_rounds=2000)


def test_plan_rounds_is_minimal_against_independent_computation():
    # Verify minimality exhaustively with the independent no-scipy computation.
    # Minimal means minimal among the n that can actually deliver the target
    # through the shipped verdict, whose acceptance region is the WINDOW
    # [c_alpha(n), max wins with S <= Tsirelson] -- not the open upper tail.
    from qcref.chsh import TSIRELSON_S, omega_to_s

    def _independent_window(n: int, c: int, p: float) -> float:
        """P[c <= Bin(n, p) <= u], u the largest win count chsh will still judge."""
        u = next(k for k in range(n, -1, -1) if omega_to_s(k / n) <= TSIRELSON_S)
        if c > u:
            return 0.0
        return _exact_upper_tail(c, n, p) - _exact_upper_tail(u + 1, n, p)

    plan = plan_rounds(0.82, alpha=0.05, power=0.9)
    for n in range(1, plan.rounds):
        c = _independent_critical_wins(n, 0.05)
        assert c > n or _independent_window(n, c, 0.82) < 0.9
    c = _independent_critical_wins(plan.rounds, 0.05)
    assert _independent_window(plan.rounds, c, 0.82) >= 0.9


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
    # The sawtooth is a property of the binomial, not of the gate, so those figures
    # stand whatever the acceptance region is. What the gate changes is which n the
    # scan may return: a win rate of 0.9 is S = 3.2 and no longer a hypothesis the
    # planner will price at all, so the sawtooth is re-checked at a physical rate.
    plan = plan_rounds(0.82, alpha=0.05, power=0.9)
    assert plan.rounds == 360
    assert plan.critical_wins == 284


def test_plan_rounds_from_s_value():
    # S=2.4 maps to omega=0.8; documented answer n=604, c=471, power ~0.9007.
    plan = plan_rounds(s_to_omega(2.4), alpha=0.05, power=0.9)
    assert plan.rounds == 604
    assert plan.critical_wins == 471
    assert plan.power == pytest.approx(0.9007, abs=5e-4)
    assert plan.S == pytest.approx(2.4)


def test_plan_rounds_monte_carlo_certification_rate():
    # In the style of qcref.selftest: simulate at the hypothesised win rate over
    # the planned n and check the empirical certification rate clears the target
    # within Monte-Carlo tolerance (3 sigma of the binomial MC error). This is the
    # documented S=2.4 plan, far enough below the Tsirelson bound that overshoot is
    # negligible; the near-bound regime is covered below.
    trials = 20_000
    plan = plan_rounds(0.8, alpha=0.05, power=0.9)
    rng = np.random.default_rng(0)
    wins = rng.binomial(plan.rounds, 0.8, size=trials)
    status = {
        int(w): chsh(int(w), plan.rounds, alpha=0.05, setting_randomness_declared=True).status
        for w in np.unique(wins)
    }
    rate = float(np.mean([status[int(w)] == CERTIFIED for w in wins]))
    tol = 3.0 * math.sqrt(plan.power * (1.0 - plan.power) / trials)
    assert rate >= 0.9 - tol
    # and the empirical rate matches the exact power, not just the target
    assert rate == pytest.approx(plan.power, abs=tol)


def test_plan_power_is_the_rate_the_shipped_verdict_certifies():
    # The cross-check above scores runs with `wins >= critical_wins`, which is only
    # half of what chsh() decides on: chsh additionally refuses a win count implying
    # S above the Tsirelson bound. Near the bound that second condition is not a
    # corner case -- at the hypothesised S=2.7 the plan promised power 0.9071 while
    # chsh() certified 0.6423 of the runs, because 0.2674 of them landed above
    # Tsirelson and came back ASSUMPTIONS_UNMET. Score the runs with the verdict
    # itself, so the planner cannot promise a certification rate it does not deliver.
    trials = 20_000
    win_rate = s_to_omega(2.7)
    plan = plan_rounds(win_rate, alpha=0.05, power=0.9)
    rng = np.random.default_rng(0)
    wins = rng.binomial(plan.rounds, win_rate, size=trials)
    status = {
        int(w): chsh(int(w), plan.rounds, alpha=0.05, setting_randomness_declared=True).status
        for w in np.unique(wins)
    }
    rate = float(np.mean([status[int(w)] == CERTIFIED for w in wins]))
    tol = 3.0 * math.sqrt(plan.power * (1.0 - plan.power) / trials)
    assert rate >= 0.9 - tol
    assert rate == pytest.approx(plan.power, abs=tol)


def test_plan_rounds_rejects_win_rate_above_the_tsirelson_bound():
    # A hypothesis no quantum device can realise is not a plan. chsh refuses a run
    # above the Tsirelson bound as ASSUMPTIONS_UNMET however small its p-value, so
    # such a hypothesis certifies only in the runs that contradict it -- at S=3.5
    # the scan sold 60 rounds at a claimed power of 0.9962 while chsh certified
    # 0.0080 of them, and that share shrinks as n grows rather than approaching the
    # target. Refuse it, exactly as a rate at or below the classical bound is.
    for p in (s_to_omega(3.5), 0.9, 1.0):
        with pytest.raises(ValueError, match="Tsirelson"):
            plan_rounds(p, alpha=0.05, power=0.9)


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
    plan = plan_rounds(0.82, alpha=0.05, power=0.9)
    text = plan.summary()
    assert "360" in text
    assert "284" in text
    assert "0.05" in text
