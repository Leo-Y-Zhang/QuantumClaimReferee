"""Coverage self-test -- the credibility centrepiece.

A referee tool must referee itself. These Monte-Carlo checks are the point of the
whole package: they demonstrate empirically that the recommended estimators are
calibrated (coverage >= nominal; false-positive rate <= alpha), and that the naive
practice minos replaces is *miscalibrated* -- it certifies violations that are not
there. Formulas are cheap; validated coverage is what earns trust.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta, binom, norm

from .adversary import MemoryLHVAdversary, default_adversaries, play_chsh_game
from .chsh import CLASSICAL_WIN, TSIRELSON_S, chsh
from .power import certification_power, critical_wins
from .status import ASSUMPTIONS_UNMET, CERTIFIED, NOT_CERTIFIED, UNDERPOWERED

__all__ = [
    "AdversarialReport",
    "binomial_interval_coverage",
    "chsh_adversarial_false_positive_rates",
    "chsh_null_false_positive_rates",
    "naive_persetting_pvalues",
]


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


def naive_persetting_pvalues(
    setting_counts: np.ndarray, setting_wins: np.ndarray
) -> np.ndarray:
    """The field-habit per-setting-correlator p-values -- FOR CONTRAST ONLY.

    Computes, per trial, the naive analysis a "S exceeds 2 by k sigma" paper runs:
    ``S_hat = E00 + E01 + E10 - E11`` from the four per-setting correlators (each
    with its own *random* denominator), a plug-in propagated standard error
    ``SE^2 = sum_xy (1 - E_xy^2) / N_xy``, and the one-sided Gaussian tail of
    ``(S_hat - 2) / SE``. In game terms ``S_hat = 2 * sum_xy omega_xy - 4`` and
    ``SE^2 = sum_xy 4 * omega_xy * (1 - omega_xy) / N_xy``.

    This estimator has no memory-robustness theorem behind it, which is why it
    lives here (as the thing the adversarial self-test attacks) and must never be
    used to certify. Degenerate cases are default-deny: a setting pair with zero
    rounds gives ``p = 1``; a zero SE gives ``p = 0`` only if ``S_hat > 2``.

    Parameters
    ----------
    setting_counts, setting_wins:
        Integer arrays of shape ``(trials, 4)``: rounds seen and rounds won per
        setting pair, e.g. from :class:`minos.adversary.AdversaryRuns`.
    """
    counts = np.asarray(setting_counts, dtype=float)
    wins = np.asarray(setting_wins, dtype=float)
    if counts.ndim != 2 or counts.shape[1] != 4:
        raise ValueError("setting_counts must have shape (trials, 4)")
    if wins.shape != counts.shape:
        raise ValueError("setting_wins must have the same shape as setting_counts")
    if not (np.isfinite(counts).all() and np.isfinite(wins).all()):
        raise ValueError("setting_counts and setting_wins must be finite")
    if (counts < 0).any() or (wins < 0).any() or (wins > counts).any():
        # Impossible tallies (e.g. 15 wins of 10 rounds) would yield a negative
        # variance and a NaN SE that the degenerate branch maps to a CONFIDENT
        # p = 0 -- a certification from unphysical data. Refuse loudly instead.
        raise ValueError("per-setting tallies must satisfy 0 <= wins <= counts")
    if (counts % 1 != 0).any() or (wins % 1 != 0).any():
        # Fractional "round counts" are equally unphysical and would otherwise
        # produce an equally confident p-value. Same discipline: refuse loudly.
        raise ValueError("per-setting tallies must be whole numbers of rounds")

    seen_all = (counts > 0).all(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = np.where(counts > 0, wins / counts, 0.0)
        s_hat = 2.0 * omega.sum(axis=1) - 4.0
        var = np.where(counts > 0, 4.0 * omega * (1.0 - omega) / counts, 0.0)
        se = np.sqrt(var.sum(axis=1))
        z = np.where(se > 0, (s_hat - 2.0) / np.where(se > 0, se, 1.0), 0.0)

    p = np.asarray(norm.sf(z), dtype=float)
    # sign-aware degenerate branch, same discipline as chsh_null_false_positive_rates
    p = np.where(se > 0, p, np.where(s_hat > 2.0, 0.0, 1.0))
    return np.where(seen_all, p, 1.0)


@dataclass(frozen=True)
class AdversarialReport:
    """One adversary's scorecard from :func:`chsh_adversarial_false_positive_rates`.

    ``fraction_certified`` is the memory-robust false-positive rate (every
    certification of an LHV adversary is false); it must sit at or below
    ``fpr_ceiling_exact``, the exact binomial ceiling
    ``P[Bin(rounds, 3/4) >= c_alpha]`` computed by the :mod:`minos.power`
    machinery. ``fpr_naive_persetting`` is the same data pushed through the naive
    per-setting sigma test (:func:`naive_persetting_pvalues`), for contrast.
    """

    adversary: str
    rounds: int
    trials: int
    alpha: float
    fraction_certified: float
    fraction_underpowered: float
    fraction_not_certified: float
    fpr_ceiling_exact: float
    fpr_naive_persetting: float
    mean_win_rate: float
    # Runs whose win rate implies S above the Tsirelson bound. chsh refuses these
    # as ASSUMPTIONS_UNMET rather than certifying them, because no quantum device
    # can produce such a value, so an adversary reaching it has revealed itself
    # rather than won. Counted separately: folding it into fraction_certified
    # would overstate the false-positive rate, and folding it into
    # fraction_not_certified would hide that the adversary was caught by physics
    # rather than by the p-value.
    fraction_assumptions_unmet: float = 0.0

    def summary(self) -> str:
        mc_sigma = (self.fpr_ceiling_exact * (1.0 - self.fpr_ceiling_exact) / self.trials) ** 0.5
        bounded = self.fraction_certified <= self.fpr_ceiling_exact + 3.0 * mc_sigma
        robust_flag = (
            "  <- bounded" if bounded else "  <- CHECK: exceeds ceiling + 3 MC sigma"
        )
        naive_flag = (
            "  <- INVALID (inflated)" if self.fpr_naive_persetting > self.alpha + 0.005 else ""
        )
        return (
            f"{self.adversary}:\n"
            f"  certified (all false) : {self.fraction_certified:.4f}  vs exact ceiling "
            f"{self.fpr_ceiling_exact:.4f} (MC sigma {mc_sigma:.4f}){robust_flag}\n"
            f"  naive per-setting     : {self.fpr_naive_persetting:.4f}{naive_flag}\n"
            f"  verdicts              : UNDERPOWERED {self.fraction_underpowered:.4f}, "
            f"NOT_CERTIFIED {self.fraction_not_certified:.4f}\n"
            f"  mean win rate         : {self.mean_win_rate:.4f}  "
            f"(classical bound {CLASSICAL_WIN:.4f})"
        )


def chsh_adversarial_false_positive_rates(
    n: int,
    *,
    alpha: float = 0.05,
    trials: int = 2_000,
    seed: int = 0,
    adversaries: tuple[MemoryLHVAdversary, ...] | None = None,
) -> dict[str, AdversarialReport]:
    """Referee a battery of memory-LHV adversaries and score every verdict.

    For each adversary in :func:`minos.adversary.default_adversaries` (or the
    supplied ones), plays ``trials`` independent ``n``-round CHSH games in which
    outcomes may depend on all past settings and outcomes, then classifies every
    run with the shipped verdict machinery:

    * ``CERTIFIED`` runs are false positives; their rate must not exceed the
      exact ceiling ``P[Bin(n, 3/4) >= c_alpha(n)]`` -- the same exact-binomial
      quantity ``minos plan`` is built on (:func:`minos.power.certification_power`
      at the classical bound), valid against *every* memory adversary by
      stochastic dominance.
    * The remaining runs must be flagged ``UNDERPOWERED`` (point estimate above
      3/4 but evidence below ``alpha``) or ``NOT_CERTIFIED``.

    The classification uses :func:`minos.power.critical_wins` -- the exact
    acceptance region of :func:`minos.chsh.chsh` -- and a subsample of every
    batch is re-checked against ``chsh()`` itself; any disagreement raises,
    so the self-test cannot silently drift from the verdict machinery.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if trials <= 0:
        raise ValueError("trials must be a positive integer")

    c = critical_wins(n, alpha)  # validates alpha
    ceiling = certification_power(n, CLASSICAL_WIN, alpha=alpha)

    battery = tuple(adversaries) if adversaries is not None else default_adversaries()
    names = [adversary.name for adversary in battery]
    if len(set(names)) != len(names):
        # Reports are keyed by name; a duplicate would silently overwrite the
        # earlier adversary's scorecard after both batteries were paid for.
        raise ValueError(f"adversary names must be unique, got {sorted(names)}")

    reports: dict[str, AdversarialReport] = {}
    for adversary in battery:
        runs = play_chsh_game(adversary, n, trials, seed=seed)
        # Mirror chsh's acceptance region exactly, including the Tsirelson
        # condition it applies before the p-value. Keeping these in lockstep is
        # what _spot_check_against_verdict enforces below.
        unphysical = 8.0 * runs.wins > (TSIRELSON_S + 4.0) * n  # omega > ~0.8536
        certified = (runs.wins >= c) & ~unphysical
        not_certified = ~certified & ~unphysical & (4 * runs.wins <= 3 * n)
        underpowered = ~certified & ~unphysical & ~not_certified

        _spot_check_against_verdict(
            runs.wins, n, alpha, certified, not_certified, unphysical
        )

        p_naive = naive_persetting_pvalues(runs.setting_counts, runs.setting_wins)
        reports[runs.adversary] = AdversarialReport(
            adversary=runs.adversary,
            rounds=n,
            trials=trials,
            alpha=alpha,
            fraction_certified=float(np.mean(certified)),
            fraction_underpowered=float(np.mean(underpowered)),
            fraction_not_certified=float(np.mean(not_certified)),
            fraction_assumptions_unmet=float(np.mean(unphysical)),
            fpr_ceiling_exact=ceiling,
            fpr_naive_persetting=float(np.mean(p_naive <= alpha)),
            mean_win_rate=float(runs.wins.mean()) / n,
        )
    return reports


def _spot_check_against_verdict(
    wins: np.ndarray,
    n: int,
    alpha: float,
    certified: np.ndarray,
    not_certified: np.ndarray,
    unphysical: np.ndarray,
) -> None:
    """Re-run a subsample through ``chsh()`` itself: the self-test referees itself."""
    spot = min(64, wins.shape[0])
    for i in range(spot):
        status = chsh(
            int(wins[i]), n, alpha=alpha, setting_randomness_declared=True
        ).status
        if unphysical[i]:
            expected = ASSUMPTIONS_UNMET
        else:
            expected = (
                CERTIFIED
                if certified[i]
                else (NOT_CERTIFIED if not_certified[i] else UNDERPOWERED)
            )
        if status != expected:  # pragma: no cover - would be a real bug
            raise RuntimeError(
                f"selftest classification drifted from the chsh verdict at wins={int(wins[i])}, "
                f"n={n}: chsh says {status}, selftest says {expected}"
            )
