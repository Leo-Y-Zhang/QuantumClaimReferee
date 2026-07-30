"""minos -- an honest, finite-sample statistical referee for quantum claims.

Point it at raw measurement counts and get a default-deny verdict on whether a
fidelity or entanglement/CHSH claim survives finite-sample scrutiny. This is a
*measurement referee*: it does not run circuits, mitigate error, or improve results.

v1 is deliberately scoped to what can be made provably correct -- linear functionals
(per-outcome probabilities, fidelity to a fixed target), memory-robust CHSH p-values,
multiple-comparison discipline, and a coverage self-test. Full-state confidence
regions, purity, general entanglement witnesses and SPAM modelling are intentionally
out of scope and documented as such rather than approximated.
"""

from __future__ import annotations

from ._version import __version__
from .chsh import (
    CHSHResult,
    chsh,
    game_tail_pvalue,
    omega_to_s,
    s_to_omega,
    wins_from_setting_counts,
)
from .counts import CountsDataset
from .fidelity import fidelity_to_basis_state, probability_interval
from .intervals import Interval, clopper_pearson_interval, wilson_interval
from .multiple import benjamini_hochberg, bonferroni, holm
from .power import PlanResult, certification_power, critical_wins, plan_rounds
from .report import referee_report
from .verdict import Hypothesis, HypothesisResult, Study, Verdict

__all__ = [
    "__version__",
    "CHSHResult",
    "CountsDataset",
    "Hypothesis",
    "HypothesisResult",
    "Interval",
    "PlanResult",
    "Study",
    "Verdict",
    "benjamini_hochberg",
    "bonferroni",
    "certification_power",
    "chsh",
    "clopper_pearson_interval",
    "critical_wins",
    "fidelity_to_basis_state",
    "game_tail_pvalue",
    "holm",
    "omega_to_s",
    "plan_rounds",
    "probability_interval",
    "referee_report",
    "s_to_omega",
    "wilson_interval",
    "wins_from_setting_counts",
]
