"""Fidelity and per-outcome probabilities -- v1's linear functionals only.

We restrict v1 to *linear* functionals of the state, whose finite-sample uncertainty
is an honest binomial interval. The fidelity of a measured state to a fixed
computational-basis target ``|b>`` equals the probability of measuring ``b`` in the
computational basis -- a sample proportion -- so it inherits a rigorous Wilson or
Clopper-Pearson interval directly.

Fidelity to a *superposition* target (GHZ, Bell, ...) is a genuinely different, harder
problem (direct fidelity estimation across several measurement bases, or full
tomography with confidence regions over the density-matrix cone). That is deliberately
out of scope for v1 and documented rather than approximated -- issuing a shaky number
there would defeat the entire point of an honest referee.
"""

from __future__ import annotations

import dataclasses

from .counts import CountsDataset
from .intervals import Interval, clopper_pearson_interval, wilson_interval

__all__ = ["probability_interval", "fidelity_to_basis_state"]

_METHODS = {"wilson": wilson_interval, "clopper-pearson": clopper_pearson_interval}


def probability_interval(
    dataset: CountsDataset,
    bitstring: str,
    *,
    setting: str = "default",
    method: str = "wilson",
    level: float = 0.95,
) -> Interval:
    """Confidence interval for the probability of outcome ``bitstring`` in ``setting``."""
    povm = dataset.require_povm()
    if method not in _METHODS:
        raise ValueError(f"method must be one of {sorted(_METHODS)}")
    if setting not in dataset.settings:
        raise ValueError(f"unknown setting {setting!r}")
    hist = dataset.settings[setting]
    n = sum(hist.values())
    k = hist.get(bitstring, 0)
    base = _METHODS[method](k, n, level)
    return dataclasses.replace(base, assumptions=base.assumptions + (f"POVM={povm}",))


def fidelity_to_basis_state(
    dataset: CountsDataset,
    bitstring: str,
    *,
    setting: str = "default",
    method: str = "wilson",
    level: float = 0.95,
) -> Interval:
    """Fidelity to the computational-basis target ``|bitstring>``.

    This equals the probability of measuring ``bitstring`` in the computational basis,
    so it is a linear functional with an honest binomial interval. Use this only when
    the target is a basis state; superposition targets need DFE/tomography (not in v1).
    """
    interval = probability_interval(
        dataset, bitstring, setting=setting, method=method, level=level
    )
    return dataclasses.replace(
        interval, assumptions=interval.assumptions + ("target=computational_basis_state",)
    )
