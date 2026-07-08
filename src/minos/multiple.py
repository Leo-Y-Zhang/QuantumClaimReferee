"""Multiple-comparison corrections -- the "deflate the best-of-N" discipline.

When you scan many witnesses, inequalities, or qubit pairs and report the most
significant one, the reported p-value must be corrected for the search, exactly as a
best-of-N backtest Sharpe must be deflated. Quantum experiment papers rarely do this;
it is one of minos's genuinely-absent contributions. All methods are standard.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bonferroni", "holm", "benjamini_hochberg"]


def _as_array(pvals) -> np.ndarray:
    # asarray (not list()) so a scalar becomes a 0-d array and fails the ndim check
    # below with a ValueError, rather than a bare TypeError.
    arr = np.asarray(pvals, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("pvals must be a non-empty 1-D sequence")
    if np.any((arr < 0.0) | (arr > 1.0)) or np.any(~np.isfinite(arr)):
        raise ValueError("every p-value must be finite and in [0, 1]")
    return arr


def bonferroni(pvals) -> list[float]:
    """Bonferroni-adjusted p-values (control the family-wise error rate)."""
    arr = _as_array(pvals)
    return list(np.minimum(arr * arr.size, 1.0))


def holm(pvals) -> list[float]:
    """Holm step-down adjusted p-values -- uniformly more powerful than Bonferroni
    while controlling the same family-wise error rate (Holm, 1979)."""
    arr = _as_array(pvals)
    m = arr.size
    order = np.argsort(arr)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * arr[idx])
        adj[idx] = min(running, 1.0)
    return list(adj)


def benjamini_hochberg(pvals) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (control the false-discovery rate,
    1995) -- the right tool when scanning many hypotheses and tolerating a
    controlled fraction of false positives rather than any at all."""
    arr = _as_array(pvals)
    m = arr.size
    order = np.argsort(arr)
    ranks = np.arange(1, m + 1)
    scaled = arr[order] * m / ranks
    # enforce monotonicity from the largest p-value downward
    running = np.minimum.accumulate(scaled[::-1])[::-1]
    adj = np.empty(m, dtype=float)
    adj[order] = np.minimum(running, 1.0)
    return list(adj)
