"""The default-deny referee: bundle hypotheses, correct for the search, judge.

A ``Study`` collects one or more hypotheses (each a name and a valid p-value),
applies a multiple-comparison correction across them, and returns a ``Verdict`` whose
overall classification is default-deny: ``CERTIFIED`` only when *every* hypothesis
clears its bar after correction; ``ASSUMPTIONS_UNMET`` if any hypothesis could not
even be tested; otherwise ``NOT_CERTIFIED``. The verdict always sits *on top of* the
per-hypothesis numbers -- it is a headline, never a black box.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .multiple import benjamini_hochberg, bonferroni, holm

__all__ = ["Hypothesis", "HypothesisResult", "Verdict", "Study"]

_CORRECTIONS = {
    "holm": holm,
    "bonferroni": bonferroni,
    "benjamini-hochberg": benjamini_hochberg,
    "none": lambda ps: list(ps),
}

CERTIFIED = "CERTIFIED"
UNDERPOWERED = "UNDERPOWERED"
ASSUMPTIONS_UNMET = "ASSUMPTIONS_UNMET"
NOT_CERTIFIED = "NOT_CERTIFIED"


@dataclass(frozen=True)
class Hypothesis:
    name: str
    pvalue: float
    estimate: str = ""
    assumptions_met: bool = True
    detail: str = ""


@dataclass(frozen=True)
class HypothesisResult:
    name: str
    raw_p: float
    adjusted_p: float
    status: str
    estimate: str = ""
    detail: str = ""


@dataclass(frozen=True)
class Verdict:
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
            adj = "n/a" if r.raw_p != r.raw_p else f"{r.adjusted_p:.3e}"  # NaN-safe
            lines.append(
                f"  [{r.status:<16}] {r.name}: raw p={r.raw_p:.3e}, adj p={adj}"
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
        if assumptions_met and not (0.0 <= pvalue <= 1.0):
            raise ValueError("a testable hypothesis needs a p-value in [0, 1]")
        self._hypotheses.append(
            Hypothesis(name, pvalue, estimate, assumptions_met, detail)
        )
        return self

    def add_hypothesis(self, hypothesis: Hypothesis) -> "Study":
        self._hypotheses.append(hypothesis)
        return self

    def run(self) -> Verdict:
        if not self._hypotheses:
            raise ValueError("no hypotheses added")

        testable = [h for h in self._hypotheses if h.assumptions_met]
        adjusted: dict[int, float] = {}
        if testable:
            adj_vals = _CORRECTIONS[self.correction]([h.pvalue for h in testable])
            adjusted = {id(h): a for h, a in zip(testable, adj_vals)}

        results: list[HypothesisResult] = []
        for h in self._hypotheses:
            if not h.assumptions_met:
                status, raw_p, adj_p = ASSUMPTIONS_UNMET, float("nan"), float("nan")
            else:
                adj_p = adjusted[id(h)]
                raw_p = h.pvalue
                status = CERTIFIED if adj_p <= self.alpha else UNDERPOWERED
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
