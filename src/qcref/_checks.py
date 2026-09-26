"""Input guards shared across modules: a tally is a whole number, never a flag.

Comparisons such as ``0 <= wins <= rounds`` are satisfied by infinity and by
fractions, and ``bool`` is a subclass of ``int``; each of these once reached a
verdict. The helpers here refuse them at the entry points.
"""

from __future__ import annotations

import math
import numbers

import numpy as np


def whole_number(value, what: str) -> int:
    """Return ``value`` as an ``int`` if it is a finite whole number, else raise.

    Integers (Python or numpy) pass, as do integral floats such as ``80.0``; booleans,
    fractions, ``nan`` and ``inf`` raise ``ValueError``.
    """
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{what} must be an integer count, not a boolean ({value!r})")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        f = float(value)
        if math.isfinite(f) and f.is_integer():
            return int(f)
    raise ValueError(f"{what} must be a finite integer count, got {value!r}")


def is_boolean(value) -> bool:
    return isinstance(value, bool | np.bool_)
