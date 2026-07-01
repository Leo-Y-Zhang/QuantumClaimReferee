"""CHSH / Bell-inequality certification with *valid* finite-sample p-values.

The flagship of qreferee. The routine practice of "S exceeds 2 by many sigma"
(a Gaussian on the CHSH value using the observed variance) is not merely loose --
it is miscalibrated: it certifies data it should not (see :mod:`qreferee.selftest`).

We work in the CHSH *game* picture, which makes a rigorous p-value elementary:

* each round the referee draws inputs ``(x, y)`` uniformly at random;
* the players win iff ``a XOR b == x AND y``;
* under local realism the per-round win probability is at most ``3/4`` *regardless
  of the history of previous rounds*;
* therefore the total number of wins is stochastically dominated by
  ``Binomial(n, 3/4)`` -- so ``P[Binomial(n, 3/4) >= wins]`` is a valid p-value
  even under the memory loophole.

The p-value implemented here is exactly that stochastic-dominance / martingale tail
(Gill, quant-ph/0301059; Bierhorst, J. Phys. A 48 025301). The near-optimal p-values
of Elkouss & Wehner (npj QI 2016) are a tighter refinement of the same idea and are
*not* implemented here; the plain game tail is valid and conservative.

The CHSH value maps to the win probability by ``S = 8*omega - 4`` (``S = 2`` is the
local bound ``omega = 3/4``; Tsirelson's bound ``S = 2*sqrt(2)`` is ``omega ~ 0.8536``).

This module requires you to *declare* that measurement settings were randomised per
round; without that declaration the game bound does not apply and the verdict is
``ASSUMPTIONS_UNMET`` (default-deny), never a silent pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from scipy.stats import binom, norm

from .intervals import Interval, wilson_interval
from .status import ASSUMPTIONS_UNMET, CERTIFIED, NOT_CERTIFIED, UNDERPOWERED

__all__ = ["CHSHResult", "chsh", "wins_from_setting_counts", "CLASSICAL_WIN", "TSIRELSON_S"]

CLASSICAL_WIN = 0.75
TSIRELSON_S = 2.0 * math.sqrt(2.0)


def _omega_to_s(omega: float) -> float:
    return 8.0 * omega - 4.0


@dataclass(frozen=True)
class CHSHResult:
    """Outcome of a CHSH certification.

    ``p_memory_robust`` is the one to trust and the one the ``status`` is based on.
    ``p_naive_observed`` is reported only so the gap to the naive practice is visible;
    it must not be used to certify anything.
    """

    wins: int
    rounds: int
    omega: float
    S: float
    S_ci: Interval
    p_memory_robust: float
    p_azuma: float
    p_naive_observed: float
    alpha: float
    status: str
    assumptions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def certified(self) -> bool:
        return self.status == CERTIFIED

    def summary(self) -> str:
        return (
            f"CHSH: S = {self.S:.3f}  {self.S_ci}\n"
            f"  status              : {self.status}\n"
            f"  p (memory-robust)   : {self.p_memory_robust:.3e}   <- use this\n"
            f"  p (Azuma, loose)    : {self.p_azuma:.3e}\n"
            f"  p (naive, observed) : {self.p_naive_observed:.3e}   (for contrast only)\n"
            f"  assumptions         : {', '.join(self.assumptions) or 'none declared'}"
        )


def chsh(
    wins: int,
    rounds: int,
    *,
    level: float = 0.95,
    alpha: float = 0.05,
    setting_randomness_declared: bool = False,
    no_signaling: bool = True,
) -> CHSHResult:
    """Certify a CHSH violation from ``wins`` game-wins out of ``rounds`` rounds.

    Parameters
    ----------
    wins, rounds:
        Number of CHSH-game wins and total rounds (``0 <= wins <= rounds``).
    level:
        Confidence level (in ``(0, 1)``) for the reported interval on ``S``.
    alpha:
        Significance threshold (in ``(0, 1)``) for the certification decision.
    setting_randomness_declared:
        You must affirm that inputs ``(x, y)`` were chosen randomly each round.
        If ``False`` the game bound is invalid and the status is ``ASSUMPTIONS_UNMET``.
    no_signaling:
        Recorded as an assumption; the standard Bell-test spatial-separation premise.

    Returns
    -------
    CHSHResult
        With the memory-robust p-value, the interval on ``S``, and a default-deny
        ``status`` in {CERTIFIED, UNDERPOWERED, NOT_CERTIFIED, ASSUMPTIONS_UNMET}.
    """
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if not (0 <= wins <= rounds):
        raise ValueError(f"wins={wins} must satisfy 0 <= wins <= rounds={rounds}")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0, 1)")

    omega = wins / rounds
    S = _omega_to_s(omega)

    om_ci = wilson_interval(wins, rounds, level)
    S_ci = Interval(
        _omega_to_s(om_ci.point),
        _omega_to_s(om_ci.lo),
        _omega_to_s(om_ci.hi),
        level,
        "wilson->S",
        rounds,
        ("binomial",),
    )

    # Rigorous, memory-robust p-value: P[Binomial(n, 3/4) >= wins].
    p_memory_robust = float(binom.sf(wins - 1, rounds, CLASSICAL_WIN))

    # Closed-form Azuma/Hoeffding bound: valid but loose.
    delta = omega - CLASSICAL_WIN
    p_azuma = float(math.exp(-2.0 * rounds * delta * delta)) if delta > 0 else 1.0

    # Naive Gaussian on the win rate with the (anti-conservative) observed SE.
    # Reported ONLY to expose the gap; never used for the decision.
    var_obs = omega * (1.0 - omega)
    if var_obs > 0:
        z_obs = (omega - CLASSICAL_WIN) / math.sqrt(var_obs / rounds)
        p_naive_observed = float(norm.sf(z_obs))
    else:
        p_naive_observed = 0.0 if omega > CLASSICAL_WIN else 1.0

    assumptions: list[str] = []
    if setting_randomness_declared:
        assumptions.append("settings_randomised_per_round")
    if no_signaling:
        assumptions.append("no_signaling")

    if not setting_randomness_declared:
        status = ASSUMPTIONS_UNMET
    elif p_memory_robust <= alpha:
        status = CERTIFIED
    elif omega <= CLASSICAL_WIN:
        # point estimate at or below the local bound: decisive non-violation
        status = NOT_CERTIFIED
    else:
        # on the violating side but the evidence does not clear alpha: get more shots
        status = UNDERPOWERED

    return CHSHResult(
        wins=wins,
        rounds=rounds,
        omega=omega,
        S=S,
        S_ci=S_ci,
        p_memory_robust=p_memory_robust,
        p_azuma=p_azuma,
        p_naive_observed=p_naive_observed,
        alpha=alpha,
        status=status,
        assumptions=tuple(assumptions),
    )


def wins_from_setting_counts(
    counts: dict[tuple[int, int], dict[tuple[int, int], int]]
) -> tuple[int, int]:
    """Reduce per-setting outcome counts to ``(wins, rounds)`` for the CHSH game.

    ``counts`` maps each input pair ``(x, y)`` in ``{0,1}^2`` to a dict of measured
    outcome pairs ``(a, b)`` in ``{0,1}^2`` and their shot counts. A round is a win
    iff ``a XOR b == x AND y``.

    .. warning::
       The returned ``(wins, rounds)`` is only a valid *memory-robust* input when the
       inputs ``(x, y)`` were sampled from a trusted RNG independently each round.
       Pooling fixed or deterministic per-setting blocks discards round order and
       breaks the martingale premise (Gill, 2003; Bierhorst, 2015) -- do **not** then
       affirm ``setting_randomness_declared=True`` for such block designs.
    """
    wins = 0
    rounds = 0
    for (x, y), outcomes in counts.items():
        if x not in (0, 1) or y not in (0, 1):
            raise ValueError("settings (x, y) must be in {0, 1}")
        for (a, b), c in outcomes.items():
            if a not in (0, 1) or b not in (0, 1):
                raise ValueError("outcomes (a, b) must be in {0, 1}")
            if c < 0:
                raise ValueError("counts must be non-negative")
            rounds += c
            if (a ^ b) == (x & y):
                wins += c
    if rounds == 0:
        raise ValueError("no rounds supplied")
    return wins, rounds
