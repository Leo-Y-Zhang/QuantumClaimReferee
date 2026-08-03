"""Exact finite-sample power analysis for the CHSH certification: ``qcref plan``.

Answers the design question *before* the experiment: given a hypothesised per-round
win probability (or CHSH ``S`` value), how many rounds are needed so that the shipped
certification -- the game-tail criterion ``P[Bin(n, 3/4) >= wins] <= alpha`` of
:func:`qcref.chsh.chsh` -- succeeds with at least the target probability?

Everything here is exact Binomial; there is no Gaussian shortcut anywhere. The
threshold is *derived from* :func:`qcref.chsh.game_tail_pvalue`, the very function the
verdict logic uses, so the plan and the verdict can never disagree.

The subtlety that justifies the module: exact binomial power is **non-monotone in n**
(a sawtooth). The critical win count ``c_alpha(n)`` is an integer, so as ``n`` grows
the power climbs until ``c_alpha`` steps up by one, at which point the power *drops*,
then climbs again. Consequently a bisection over ``n`` -- which assumes monotone
power -- is simply wrong: it can probe an ``n`` inside a dip and search upward past
the true minimum. :func:`plan_rounds` therefore scans ``n`` upward and returns the
first (hence minimal) ``n`` whose exact power meets the target. Note the flip side:
because of the same sawtooth, some *larger* ``n`` may again fall short of the target;
the returned ``n`` is the minimal one that meets it, exactly as specified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import binom

from .chsh import CLASSICAL_WIN, TSIRELSON_S, game_tail_pvalue, omega_to_s

__all__ = [
    "PlanResult",
    "certification_power",
    "critical_wins",
    "is_physically_attainable",
    "minimum_physical_rounds",
    "plan_rounds",
]

DEFAULT_MAX_ROUNDS = 1_000_000


def _validate_alpha(alpha: float) -> None:
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")


def critical_wins(rounds: int, alpha: float) -> int:
    """The critical win count ``c_alpha(n)``: the smallest ``wins`` that certifies.

    That is, the smallest ``k`` with ``game_tail_pvalue(k, rounds) <= alpha`` -- the
    exact acceptance region of :func:`qcref.chsh.chsh`. Returns ``rounds + 1`` when
    even winning every round does not certify (``0.75**rounds > alpha``), i.e. the
    certification is unattainable at this ``n``.
    """
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    _validate_alpha(alpha)
    # Seed with scipy's inverse survival function, then pin down the boundary with
    # the shipped p-value itself so the answer is defined by the verdict criterion,
    # not by isf edge-case behaviour.
    k = int(binom.isf(alpha, rounds, CLASSICAL_WIN)) + 1
    k = max(1, min(k, rounds + 1))
    while k > 1 and game_tail_pvalue(k - 1, rounds) <= alpha:
        k -= 1
    while k <= rounds and game_tail_pvalue(k, rounds) > alpha:
        k += 1
    return k


def is_physically_attainable(rounds: int, alpha: float) -> bool:
    """Can the critical win count at ``rounds`` be reached by a quantum device?

    :func:`critical_wins` answers an arithmetic question and cannot see physics.
    At small ``n`` its answer implies a CHSH value above the Tsirelson bound,
    which no quantum system can produce: at ``alpha=0.05``, ``n=11`` demands 11
    wins out of 11 (``S = 4.000``, the algebraic PR-box maximum) and ``n=40``
    demands 35 (``S = 3.000``). Those thresholds are not merely demanding, they
    are unreachable, so planning such a run is advice that cannot be followed
    honestly.
    """
    c = critical_wins(rounds, alpha)
    if c > rounds:
        return False
    return omega_to_s(c / rounds) <= TSIRELSON_S


def minimum_physical_rounds(alpha: float, *, max_rounds: int = DEFAULT_MAX_ROUNDS) -> int:
    """Smallest ``n`` whose certification threshold a quantum device could meet.

    60 at ``alpha=0.05``, 101 at ``0.01``, 169 at ``0.001``. Below this a run
    cannot certify without reporting a physically impossible value, so
    :func:`plan_rounds` never proposes one.
    """
    _validate_alpha(alpha)
    for n in range(1, max_rounds + 1):
        if is_physically_attainable(n, alpha):
            return n
    raise ValueError(
        f"no round count up to {max_rounds} has a physically attainable "
        f"certification threshold at alpha={alpha}"
    )


def certification_power(rounds: int, win_rate: float, *, alpha: float = 0.05) -> float:
    """Exact probability that ``chsh`` certifies at ``alpha`` with ``rounds`` rounds.

    Assumes rounds are i.i.d. wins with probability ``win_rate``; the value is the
    exact binomial tail ``P[Bin(rounds, win_rate) >= c_alpha(rounds)]``. Zero when
    certification is unattainable at this ``rounds``.
    """
    if not (math.isfinite(win_rate) and 0.0 <= win_rate <= 1.0):
        raise ValueError("win_rate must be in [0, 1]")
    c = critical_wins(rounds, alpha)
    if c > rounds:
        return 0.0
    return float(binom.sf(c - 1, rounds, win_rate))


@dataclass(frozen=True)
class PlanResult:
    """Outcome of :func:`plan_rounds`: the minimal experiment that meets the target.

    ``power`` is the exact certification probability at ``rounds`` (it meets
    ``target_power``); ``critical_wins`` is the win count that must be reached.
    """

    rounds: int
    critical_wins: int
    power: float
    target_power: float
    win_rate: float
    S: float
    alpha: float

    def summary(self) -> str:
        return (
            f"PLAN: {self.rounds} rounds  (hypothesis: win rate {self.win_rate:.4f}, "
            f"S = {self.S:.3f})\n"
            f"  certify iff wins >=  : {self.critical_wins}  (alpha={self.alpha})\n"
            f"  exact power          : {self.power:.4f}  (target {self.target_power})\n"
            f"  note                 : minimal n meeting the target; exact binomial\n"
            f"                         power is sawtoothed, so a larger n can dip\n"
            f"                         below the target again"
        )


def plan_rounds(
    win_rate: float,
    *,
    alpha: float = 0.05,
    power: float = 0.9,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> PlanResult:
    """The minimal ``n`` with ``P[certify at alpha with n rounds] >= power``.

    Parameters
    ----------
    win_rate:
        Hypothesised per-round win probability, strictly above the classical bound
        ``3/4`` (at or below it the certification is calibrated, so its success
        probability never exceeds ``alpha`` and no target power is reachable).
        Convert a CHSH value via :func:`qcref.chsh.s_to_omega`.
    alpha:
        Significance threshold of the certification being planned for.
    power:
        Target certification probability, in ``(0, 1)``.
    max_rounds:
        Scan cap; a ``ValueError`` is raised if the target is not met by then.

    Returns
    -------
    PlanResult
        With the minimal ``rounds``, the implied critical win count, and the exact
        power achieved there.

    Notes
    -----
    Exact binomial power is non-monotone in ``n`` (see the module docstring), so
    this scans ``n = 1, 2, ...`` and stops at the first ``n`` meeting the target --
    the only stopping rule that is correct under the sawtooth. The critical count is
    updated incrementally using the fact that ``c_alpha(n+1)`` is either
    ``c_alpha(n)`` or ``c_alpha(n) + 1``: the tail ``P[Bin(n, 3/4) >= k]`` grows
    with ``n`` at fixed ``k`` (so ``c_alpha`` cannot decrease), and a run of
    ``n + 1`` rounds with at least ``c_alpha(n) + 1`` wins contains ``n`` rounds
    with at least ``c_alpha(n)`` wins (so the step is at most one).
    """
    if not (math.isfinite(win_rate) and 0.0 <= win_rate <= 1.0):
        raise ValueError("win_rate must be in [0, 1]")
    if win_rate <= CLASSICAL_WIN:
        raise ValueError(
            f"win_rate must exceed the classical bound {CLASSICAL_WIN}: at or below "
            "it the certify probability never exceeds alpha, so no round count "
            "reaches the target power"
        )
    _validate_alpha(alpha)
    if not (math.isfinite(power) and 0.0 < power < 1.0):
        raise ValueError("power must be in (0, 1)")
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive")

    c = critical_wins(1, alpha)
    for n in range(1, max_rounds + 1):
        if n > 1 and game_tail_pvalue(c, n) > alpha:
            c += 1  # the critical count steps up by exactly one
        # Skip any n whose threshold implies S above the Tsirelson bound: the
        # plan would be arithmetically valid and physically impossible to meet.
        if c <= n and omega_to_s(c / n) <= TSIRELSON_S:
            achieved = float(binom.sf(c - 1, n, win_rate))
            if achieved >= power:
                return PlanResult(
                    rounds=n,
                    critical_wins=c,
                    power=achieved,
                    target_power=power,
                    win_rate=win_rate,
                    S=omega_to_s(win_rate),
                    alpha=alpha,
                )
    raise ValueError(
        f"target power {power} not reached within max_rounds={max_rounds} at "
        f"win_rate={win_rate}, alpha={alpha}; raise max_rounds to keep scanning"
    )
