"""Coverage self-test -- the credibility centrepiece.

A referee tool must referee itself. These Monte-Carlo checks are the point of the
whole package: they demonstrate empirically that the recommended estimators are
calibrated (coverage >= nominal; false-positive rate <= alpha), and that the naive
practice qreferee replaces is *miscalibrated* -- it certifies violations that are not
there. Formulas are cheap; validated coverage is what earns trust.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta, binom, norm

from .chsh import CLASSICAL_WIN

__all__ = ["binomial_interval_coverage", "chsh_null_false_positive_rates"]


def binomial_interval_coverage(
    method: str,
    p_true: float,
    n: int,
    *,
    level: float = 0.95,
    trials: int = 20_000,
    seed: int = 0,
) -> float:
    """Empirical coverage of a binomial-interval method at ``p_true``.

    Returns the fraction of simulated experiments whose interval contains the true
    proportion. A valid method sits at or above ``level``.
    """
    rng = np.random.default_rng(seed)
    k = rng.binomial(n, p_true, size=trials).astype(float)
    omega = k / n

    if method == "wilson":
        z = float(norm.ppf(1.0 - (1.0 - level) / 2.0))
        denom = 1.0 + z * z / n
        center = (omega + z * z / (2.0 * n)) / denom
        half = (z * np.sqrt(omega * (1.0 - omega) / n + z * z / (4.0 * n * n))) / denom
        lo = np.maximum(0.0, center - half)
        hi = np.minimum(1.0, center + half)
    elif method == "clopper-pearson":
        alpha = 1.0 - level
        lo = np.where(k == 0, 0.0, beta.ppf(alpha / 2.0, k, n - k + 1))
        hi = np.where(k == n, 1.0, beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    else:
        raise ValueError("method must be 'wilson' or 'clopper-pearson'")

    return float(np.mean((lo <= p_true) & (p_true <= hi)))


def chsh_null_false_positive_rates(
    n: int,
    *,
    alpha: float = 0.05,
    trials: int = 200_000,
    seed: int = 0,
) -> dict[str, float]:
    """False-positive rates of three CHSH tests under the i.i.d. local-realism null.

    Simulates experiments where every round wins with probability exactly ``3/4``
    (a local strategy at the classical bound) and reports the fraction each test
    wrongly certifies at ``alpha``. The rigorous game tail must be ``<= alpha``; the
    naive observed-SE Gaussian is expected to exceed it, especially at small ``n``.
    """
    rng = np.random.default_rng(seed)
    k = rng.binomial(n, CLASSICAL_WIN, size=trials)
    omega = k / n

    p_memory_robust = binom.sf(k - 1, n, CLASSICAL_WIN)

    se_obs = np.sqrt(omega * (1.0 - omega) / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        # sign-aware degenerate branch: an all-loss sample (omega=0) must map to p=1,
        # not p=0. (Only reachable at absurdly small n, but correct is correct.)
        z_obs = np.where(
            se_obs > 0,
            (omega - CLASSICAL_WIN) / se_obs,
            np.where(omega > CLASSICAL_WIN, np.inf, -np.inf),
        )
    p_naive_observed = norm.sf(z_obs)

    se_null = np.sqrt(CLASSICAL_WIN * (1.0 - CLASSICAL_WIN) / n)
    p_naive_null = norm.sf((omega - CLASSICAL_WIN) / se_null)

    return {
        "memory_robust": float(np.mean(p_memory_robust <= alpha)),
        "naive_null": float(np.mean(p_naive_null <= alpha)),
        "naive_observed": float(np.mean(p_naive_observed <= alpha)),
    }
