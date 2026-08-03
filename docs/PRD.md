# PRD — Quantum Claim Referee

**Status:** built (retrospective — the code came first; this records the intent it was built to)
**Date:** 2026-08-03 · **Repo:** QuantumClaimReferee · **Related:** [TDD](TDD.md), [App Flow](APP_FLOW.md), [Design Brief](DESIGN_BRIEF.md)

## Problem

The standard way to report a Bell/CHSH violation is "*S* exceeds 2 by *k* sigma",
using a Gaussian on the CHSH value with the observed standard error. At realistic
shot counts that test is not merely loose, it is **miscalibrated**: it certifies
violations that are not there. The shipped self-test measures a false-positive rate
of ~0.074 against a nominal 0.05 at *n* = 80 rounds under the exact local-realism
null (`qcref selftest --n 80`), and the adversarial battery pushes the per-setting
variant of the same habit to **0.2953 at n = 80** — five to six times its nominal
rate — against a genuinely classical player (`qcref selftest --adversarial`). The
inflation does not wash out with more data: 0.2675 at *n* = 4000.

Two further habits compound it. Scanning many qubit pairs or witnesses and
reporting the best one, with no multiple-comparison correction. And reporting a
number without recording the measurement model it assumes, so a reader cannot tell
what was assumed versus measured.

The problem is real but the population that hits it is narrow: it bites hardest in
the **low-shot, many-hypotheses** regime. At *n* = 8000 every method here agrees and
the tool earns nothing. That boundary is stated in the README rather than hidden.

## Who it is for

Someone who has already run the experiment and now has to decide whether the number
they are about to publish survives scrutiny — a physicist or a reviewer holding
`{bitstring: shots}` histograms or a CHSH win count, working at a few hundred to a
few thousand rounds. Secondarily: this repository is a portfolio artefact, and the
judgement on display (killing the wrong idea, scoping v1 to what could be proved) is
part of what it is for. `NOTES.md` says plainly that as a business it is a
shelve-it. That honesty is a requirement, not a caveat.

## Success looks like

- [x] A CHSH certification whose false-positive rate stays at or below α under the
      exact local-realism null, measured, not argued — `qcref selftest`.
- [x] The same guarantee holds against **history-dependent** local players (the
      memory loophole), demonstrated by simulation against four adversaries rather
      than cited from a paper — `qcref selftest --adversarial`.
- [x] Every certification that is not clearly earned is refused, and the refusal
      says which of the four statuses applies and why.
- [x] A report that renders byte-identical from identical inputs, carrying tool and
      dependency versions, so a number in a paper can be traced to what produced it.
- [x] Exit code 0 iff CERTIFIED, so the tool drops into a CI gate like a test runner.
- [x] The tool's own limits are written down where a user will read them, not
      buried: v1 does linear functionals and CHSH, and nothing else.

## Requirements

**Must**
- Memory-robust CHSH p-value `P[Bin(n, 3/4) ≥ wins]` as the *only* certification
  criterion, with a single implementation shared by the verdict, the power analysis
  and the self-test, so they cannot disagree.
- Default-deny: an undeclared POVM, undeclared setting randomisation, or an
  unphysical value is a refusal, never a guess and never a silent pass.
- Multiple-comparison correction (Holm, Benjamini–Hochberg, Bonferroni) across a
  bundle of hypotheses, with the overall verdict defaulting to deny.
- Confidence intervals that behave near 0 and 1 (Wilson, Clopper–Pearson), because
  near-pure states are the common case. No Wald.
- A coverage self-test shipped in the package and run in CI. A referee that cannot
  referee itself has no business refereeing anyone else.
- Offline. No network access, ever.

**Should**
- Exact-binomial power analysis (`qcref plan`) so a run can be sized before it is
  paid for, using the same critical-win threshold the verdict uses.
- Framework-agnostic ingestion (`CountsDataset`) so Qiskit / Cirq / PennyLane counts
  all reduce to one schema, keeping the dependency footprint at numpy + scipy.

**Won't (this time)**
- Full density-matrix confidence *regions* over the PSD cone.
- Purity, Rényi entropy, negativity, general entanglement witnesses.
- Fidelity to a superposition target (needs DFE or tomography).
- SPAM / readout-error modelling.
- Optional-stopping robustness: choosing *n* adaptively is a different loophole and
  the fixed-*n* game tail does not cover it.

## Explicitly out of scope

**This is not a physics engine.** It does not run circuits, mitigate error, or
improve results. It is the statistics layer that sits on top of one, and every
number it produces is conditional on a measurement model the *user* declares.

The five "Won't" items above are out of scope for one reason, stated in `NOTES.md`
and worth repeating: they need genuine quantum-information depth, and a solo author
should not self-certify a PSD-cone confidence region. Shipping a shaky number there
would destroy the only thing this tool sells — that its numbers can be trusted.
They are documented as absent rather than approximated.

Also out of scope: any claim about *whether the device is good*. The tool answers
"does this claim survive finite-sample scrutiny", not "is this a good qubit".

## Safety and privacy

- **Personal data: none.** Inputs are integer shot counts and win tallies. There is
  no account, no store, no telemetry, and no network call anywhere in the package —
  the dependency set is numpy and scipy.
- **Access control: none, and none is needed.** It is a local library and CLI with
  no server, no database and no shared state. There is nothing to revoke, so the
  usual revocation question does not arise; if this ever grows a hosted surface,
  that answer changes and this section must be rewritten before it ships.
- **Worst outcome if it is wrong:** a false CERTIFIED that a user cites in a paper.
  That is why the failure direction is asymmetric by design — every guard fails
  closed (refuse), never open (certify). Concretely: an undeclared POVM refuses to
  estimate; an undeclared randomisation refuses to certify; an *S* above the
  Tsirelson bound is treated as evidence the data are not what they are claimed to
  be, not as a stronger result; an incomplete four-setting run raises rather than
  pooling (an early version would have certified a purely classical device from a
  single setting at *p* = 0); an invalid p-value raises at the point it is added.
- **Second-order risk:** a user reads the naive contrast p-value as a result. It is
  printed only to show the gap, and is labelled `(for contrast only)` on every line
  it appears on, in the summary, the self-test and the docstrings.

## Open questions

- Is the Elkouss–Wehner (npj QI 2016) near-optimal p-value worth implementing? It is
  a tighter refinement of the same idea; the plain game tail is valid and
  conservative, so this is a power improvement, not a correctness fix. Unresolved,
  and the README says so.
- Optional stopping: is a sequential/anytime-valid variant (e-values, test
  martingales) the right v2, given that adaptive *n* is the loophole most likely to
  be hit accidentally by a real lab?
- Nothing here blocks v1. Both would change v2's design.

## Not doing / rejected alternatives

- **Repurposing the finance overfitting tool** (deflated Sharpe, PBO, purged
  walk-forward) at quantum data. Rejected after reading: the machinery does not
  transfer — quantum data has a *known* Born-rule null and i.i.d. resampleable
  shots, which is exactly what the finance methods exist to work around. What
  transferred was the ethos, not a line of code. `NOTES.md` is the long version.
- **"Solving" decoherence / interference.** Rejected as incoherent: interference is
  the resource, not the defect, and the unwanted effects are hardware and
  error-correction problems, not statistics problems.
- **Bootstrap or Wald intervals.** Rejected: Wald overshoots [0, 1] and collapses to
  zero width as the proportion approaches 1, precisely the near-pure-state regime
  this tool is used in.
- **Azuma–Hoeffding as the headline p-value.** Implemented and reported, but not the
  criterion: it is valid and much looser (4.066e-01 vs 7.399e-02 on the worked
  example). Kept visible so the conservatism of the choice is auditable.
- **Bisection over *n* in the power analysis.** Rejected because exact binomial power
  is non-monotone in *n* (a sawtooth), so bisection can land in a dip and return a
  wrong answer. A linear scan is slower and correct.
- **Publishing to PyPI.** Not done. The licence is proprietary, source-available,
  portfolio-viewing only, and an unmaintained statistics package that people
  might cite is worse than no package.
