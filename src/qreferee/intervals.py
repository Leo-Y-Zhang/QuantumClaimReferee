"""Finite-sample confidence intervals for binomial proportions.

Everything qreferee v1 measures is a *linear functional* of the quantum state --
a per-outcome probability, or a fidelity to a fixed target expressed as an average
of bounded measurement outcomes. Such quantities are estimated as a sample
proportion, so honest error bars reduce to a binomial confidence interval.

We deliberately provide the Wilson score interval (the sensible default) and the
Clopper-Pearson exact interval, and deliberately *avoid* the Wald interval, which
overshoots [0, 1] and collapses to zero width as the proportion approaches 0 or 1 --
precisely the near-pure-state regime common in quantum experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from scipy.stats import beta, norm

__all__ = ["Interval", "wilson_interval", "clopper_pearson_interval"]


@dataclass(frozen=True)
class Interval:
    """A self-describing confidence interval.

    An interval carries not just the numbers but the ``method`` used, the sample
    size ``n`` it rests on, and the ``assumptions`` it is only valid under, so a
    downstream verdict (or a human referee) can never mistake its scope.
    """

    point: float
    lo: float
    hi: float
    level: float
    method: str
    n: int
    assumptions: tuple[str, ...] = field(default_factory=tuple)

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi

    def __str__(self) -> str:
        return (
            f"{self.point:.4f}  [{self.lo:.4f}, {self.hi:.4f}]  "
            f"({self.level * 100:g}% {self.method}, n={self.n})"
        )


def _validate(k: int, n: int, level: float) -> None:
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if not (0 <= k <= n):
        raise ValueError(f"k={k} must satisfy 0 <= k <= n={n}")
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0, 1)")


def wilson_interval(k: int, n: int, level: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion ``k / n``.

    Well behaved for small ``n`` and for proportions near 0 or 1, where the naive
    Wald interval fails. This is the recommended default.
    """
    _validate(k, n, level)
    z = float(norm.ppf(1.0 - (1.0 - level) / 2.0))
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return Interval(phat, lo, hi, level, "wilson", n, ("binomial",))


def clopper_pearson_interval(k: int, n: int, level: float = 0.95) -> Interval:
    """Clopper-Pearson *exact* interval (guaranteed >= nominal coverage).

    Conservative by construction -- the honest, default-deny choice when you would
    rather under-claim than over-claim significance.
    """
    _validate(k, n, level)
    alpha = 1.0 - level
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return Interval(k / n, lo, hi, level, "clopper-pearson", n, ("binomial", "exact"))
