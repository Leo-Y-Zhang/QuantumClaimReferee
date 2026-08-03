# Quantum Claim Referee — what it certifies, and what it refuses

Recorded after the fact: the code came first, and this is the intent it was built
to. [TDD](TDD.md) · [App Flow](APP_FLOW.md) · [Design Brief](DESIGN_BRIEF.md)

## The miscalibration, measured

The standard way to report a Bell/CHSH violation is "*S* exceeds 2 by *k*
sigma", using a Gaussian on the CHSH value with the observed standard error. At
realistic shot counts that test is not merely loose. It is **miscalibrated**: it
certifies violations that are not there.

The shipped self-test measures a false-positive rate of about 0.074 against a
nominal 0.05 at *n* = 80 rounds under the exact local-realism null
(`qcref selftest --n 80`). The adversarial battery pushes the per-setting variant
of the same habit to **0.2953 at n = 80** — five to six times its nominal rate —
against a genuinely classical player (`qcref selftest --adversarial`). And the
inflation does not wash out with more data: 0.2675 at *n* = 4000.

Two further habits compound it. Scanning many qubit pairs or witnesses and
reporting the best one, with no multiple-comparison correction. And reporting a
number without recording the measurement model it assumes, so a reader cannot
tell what was assumed from what was measured.

## Where it earns nothing

The problem is real, but the population that hits it is narrow. It bites hardest
in the **low-shot, many-hypotheses** regime. At *n* = 8000 every method here
agrees and this tool adds nothing at all.

That boundary is stated in the README rather than hidden, and it is the reason
the scope below is as small as it is.

## Requirements

**Must**

- Memory-robust CHSH p-value `P[Bin(n, 3/4) ≥ wins]` as the *only* certification
  criterion, with a single implementation shared by the verdict, the power
  analysis and the self-test, so they cannot disagree.
- Default-deny: an undeclared POVM, undeclared setting randomisation, or an
  unphysical value is a refusal — never a guess, never a silent pass.
- Multiple-comparison correction (Holm, Benjamini–Hochberg, Bonferroni) across a
  bundle of hypotheses, with the overall verdict defaulting to deny.
- Confidence intervals that behave near 0 and 1 (Wilson, Clopper–Pearson),
  because near-pure states are the common case. No Wald.
- A coverage self-test shipped in the package and run in CI. A referee that
  cannot referee itself has no business refereeing anyone else.
- Offline. No network access, ever.

**Should**

- Exact-binomial power analysis (`qcref plan`), so a run can be sized before it
  is paid for, using the same critical-win threshold the verdict uses.
- Framework-agnostic ingestion (`CountsDataset`), so Qiskit, Cirq and PennyLane
  counts all reduce to one schema, keeping the dependency footprint at numpy plus
  scipy.

**Won't, this time**

- Full density-matrix confidence *regions* over the PSD cone.
- Purity, Rényi entropy, negativity, general entanglement witnesses.
- Fidelity to a superposition target, which needs DFE or tomography.
- SPAM and readout-error modelling.
- Optional-stopping robustness. Choosing *n* adaptively is a different loophole,
  and the fixed-*n* game tail does not cover it.

## Refuse, never guess

The worst outcome is a false CERTIFIED that a user cites in a paper. Every guard
therefore fails closed, and the concrete list is short enough to hold in mind:

- An undeclared POVM refuses to estimate.
- An undeclared setting randomisation refuses to certify.
- An *S* above the Tsirelson bound is treated as evidence the data are not what
  they are claimed to be, rather than as a stronger result.
- An incomplete four-setting run raises rather than pooling. An early version
  would have certified a purely classical device from a single setting at
  *p* = 0.
- An invalid p-value raises at the point it is added.

There is a second-order risk in the other direction: a user reading the naive
contrast p-value as a result. It is printed only to show the gap, and it is
labelled `(for contrast only)` on every line it appears on — in the summary, in
the self-test, and in the docstrings.

## Marks of done

- [x] A CHSH certification whose false-positive rate stays at or below α under
      the exact local-realism null, measured rather than argued — `qcref
      selftest`.
- [x] The same guarantee holds against **history-dependent** local players, the
      memory loophole, demonstrated by simulation against four adversaries rather
      than cited from a paper — `qcref selftest --adversarial`.
- [x] Every certification that is not clearly earned is refused, and the refusal
      says which of the four statuses applies and why.
- [x] A report that renders byte-identical from identical inputs, carrying tool
      and dependency versions, so a number in a paper can be traced to what
      produced it.
- [x] Exit code 0 iff CERTIFIED, so the tool drops into a CI gate like a test
      runner.
- [x] The tool's own limits are written where a user will read them: v1 does
      linear functionals and CHSH, and nothing else.

## Who it is for, and the honest verdict on it

Someone who has already run the experiment and now has to decide whether the
number they are about to publish survives scrutiny — a physicist or a reviewer
holding `{bitstring: shots}` histograms or a CHSH win count, working at a few
hundred to a few thousand rounds.

Secondarily, this repository is a portfolio artefact, and the judgement on
display — killing the wrong idea, scoping v1 to what could be proved — is part of
what it is for. `NOTES.md` says plainly that as a business it is a shelve-it.
That honesty is a requirement, not a caveat.

## Not a physics engine

It does not run circuits, mitigate error, or improve results. It is the
statistics layer that sits on top of one, and every number it produces is
conditional on a measurement model the *user* declares.

The five "Won't" items above share one reason, stated in `NOTES.md` and worth
repeating: they need genuine quantum-information depth, and a solo author should
not self-certify a PSD-cone confidence region. Shipping a shaky number there
would destroy the only thing this tool sells — that its numbers can be trusted.
They are documented as absent rather than approximated.

Also outside the boundary: any claim about *whether the device is good*. The tool
answers "does this claim survive finite-sample scrutiny", not "is this a good
qubit".

## Privacy, briefly

No personal data. Inputs are integer shot counts and win tallies. No account, no
store, no telemetry, and no network call anywhere in the package; the dependency
set is numpy and scipy.

There is no access control and none is needed — a local library and CLI with no
server, no database and no shared state, so nothing can be revoked. If this ever
grows a hosted surface, that answer changes and this paragraph has to be
rewritten before it ships.

## Rejected

**Repurposing the finance overfitting tool** — deflated Sharpe, PBO, purged
walk-forward — at quantum data. Rejected after reading: the machinery does not
transfer, because quantum data has a *known* Born-rule null and i.i.d.
resampleable shots, which is exactly what the finance methods exist to work
around. What transferred was the ethos, not a line of code. `NOTES.md` is the
long version.

**"Solving" decoherence or interference.** Incoherent as a goal: interference is
the resource rather than the defect, and the unwanted effects are hardware and
error-correction problems, not statistics problems.

**Bootstrap or Wald intervals.** Wald overshoots [0, 1] and collapses to zero
width as the proportion approaches 1 — precisely the near-pure-state regime this
tool is used in.

**Azuma–Hoeffding as the headline p-value.** Implemented and reported, but not
the criterion: it is valid and much looser (4.066e-01 against 7.399e-02 on the
worked example). Kept visible so the conservatism of the choice is auditable.

**Bisection over *n* in the power analysis.** Exact binomial power is non-monotone
in *n* — a sawtooth — so bisection can land in a dip and return a wrong answer. A
linear scan is slower and correct.

**Publishing to PyPI.** The licence is proprietary, source-available, portfolio
viewing only, and an unmaintained statistics package that people might cite is
worse than no package.

## Two open questions

Is the Elkouss–Wehner (npj QI 2016) near-optimal p-value worth implementing? It
is a tighter refinement of the same idea, and since the plain game tail is valid
and conservative, this is a power improvement rather than a correctness fix.
Unresolved, and the README says so.

Is a sequential or anytime-valid variant — e-values, test martingales — the right
v2, given that adaptive *n* is the loophole a real lab is most likely to hit by
accident?

Neither blocks v1. Both would change v2's design.
