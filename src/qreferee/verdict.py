"""The default-deny referee: bundle hypotheses, correct for the search, judge.

A ``Study`` collects one or more hypotheses (each a name and a valid p-value),
applies a multiple-comparison correction across them, and returns a ``Verdict`` whose
overall classification is default-deny: ``CERTIFIED`` only when *every* hypothesis
clears its bar after correction; ``ASSUMPTIONS_UNMET`` if any hypothesis could not
even be tested; otherwise ``NOT_CERTIFIED``. The verdict always sits *on top of* the
per-hypothesis numbers -- it is a headline, never a black box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .multiple import benjamini_hochberg, bonferroni, holm
from .status import ASSUMPTIONS_UNMET, CERTIFIED, NOT_CERTIFIED

__all__ = ["Hypothesis", "HypothesisResult", "Verdict", "Study"]

_CORRECTIONS = {
    "holm": holm,
    "bonferroni": bonferroni,
    "benjamini-hochberg": benjamini_hochberg,
    "none": lambda ps: list(ps),
}


def _valid_p(p: float) -> bool:
    return math.isfinite(p) and 0.0 <= p <= 1.0


@dataclass(frozen=True)
class Hypothesis:
    name: str
    pvalue: float
    estimate: str = ""
    assumptions_met: bool = True
    detail: str = ""


@dataclass(frozen=True)
class HypothesisResult:
    """One row of a :class:`Verdict`: a hypothesis with its raw and adjusted p and status."""

    name: str
    raw_p: float
    adjusted_p: float
    status: str
    estimate: str = ""
    detail: str = ""


@dataclass(frozen=True)
class Verdict:
    """The default-deny outcome of a :class:`Study`."""

    classification: str
    alpha: float
    correction: str
    results: list[HypothesisResult] = field(default_factory=list)

    @property
    def certified(self) -> bool:
        return self.classification == CERTIFIED

    def summary(self) -> str:
        lines = [
            f"VERDICT: {self.classification}  "
            f"(alpha={self.alpha}, correction={self.correction})"
        ]
        for r in self.results:
            raw = "n/a" if not math.isfinite(r.raw_p) else f"{r.raw_p:.3e}"
            adj = "n/a" if not math.isfinite(r.adjusted_p) else f"{r.adjusted_p:.3e}"
            lines.append(
                f"  [{r.status:<16}] {r.name}: raw p={raw}, adj p={adj}"
                + (f"  ({r.estimate})" if r.estimate else "")
            )
        return "\n".join(lines)


class Study:
    """Accumulate hypotheses, then :meth:`run` to get a default-deny :class:`Verdict`."""

    def __init__(self, *, alpha: float = 0.05, correction: str = "holm") -> None:
        if correction not in _CORRECTIONS:
            raise ValueError(f"correction must be one of {sorted(_CORRECTIONS)}")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.correction = correction
        self._hypotheses: list[Hypothesis] = []

    def add(
        self,
        name: str,
        pvalue: float,
        *,
        estimate: str = "",
        assumptions_met: bool = True,
        detail: str = "",
    ) -> "Study":
        return self.add_hypothesis(
            Hypothesis(name, pvalue, estimate, assumptions_met, detail)
        )

    def add_hypothesis(self, hypothesis: Hypothesis) -> "Study":
        # Validate at every entry point -- a testable hypothesis MUST carry a real
        # p-value, so an invalid one can never slip through to a false CERTIFIED
        # (e.g. via correction='none', which does not re-validate).
        if hypothesis.assumptions_met and not _valid_p(hypothesis.pvalue):
            raise ValueError(
                f"hypothesis {hypothesis.name!r}: a testable hypothesis needs a finite "
                "p-value in [0, 1]"
            )
        self._hypotheses.append(hypothesis)
        return self

    def run(self) -> Verdict:
        if not self._hypotheses:
            raise ValueError("no hypotheses added")

        hyps = self._hypotheses
        testable_idx = [i for i, h in enumerate(hyps) if h.assumptions_met]
        # Defensive re-check: never correct or compare an invalid p-value.
        for i in testable_idx:
            if not _valid_p(hyps[i].pvalue):
                raise ValueError(
                    f"hypothesis {hyps[i].name!r} has an invalid p-value: {hyps[i].pvalue}"
                )
        adjusted: dict[int, float] = {}
        if testable_idx:
            adj_vals = _CORRECTIONS[self.correction]([hyps[i].pvalue for i in testable_idx])
            adjusted = dict(zip(testable_idx, adj_vals))

        results: list[HypothesisResult] = []
        for i, h in enumerate(hyps):
            if not h.assumptions_met:
                status, raw_p, adj_p = ASSUMPTIONS_UNMET, float("nan"), float("nan")
            else:
                adj_p = adjusted[i]
                raw_p = h.pvalue
                status = CERTIFIED if adj_p <= self.alpha else NOT_CERTIFIED
            results.append(
                HypothesisResult(h.name, raw_p, adj_p, status, h.estimate, h.detail)
            )

        statuses = {r.status for r in results}
        if ASSUMPTIONS_UNMET in statuses:
            classification = ASSUMPTIONS_UNMET
        elif statuses == {CERTIFIED}:
            classification = CERTIFIED
        else:
            classification = NOT_CERTIFIED

        return Verdict(classification, self.alpha, self.correction, results)
