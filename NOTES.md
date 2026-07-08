# How minos came to be (an honest origin story)

This project exists because I tried to talk myself *out* of a bad idea and found a
good one underneath it.

## The wrong question

I maintain a small statistics tool in quantitative finance that answers one question:
*is an apparent trading-strategy edge real, or an artefact of luck and searching over
many configurations?* (Deflated Sharpe ratio, probability of backtest overfitting,
purged walk-forward cross-validation.)

I wondered whether that tool could be pointed at quantum computing — validating
entanglement, or "solving" interference / decoherence / error correction.

The honest answer, after a lot of reading, was **no**, and it's worth being blunt
about why:

- Entanglement certification lives in a different mathematical universe: density
  matrices, partial transpose, negativity, entanglement witnesses, Born-rule
  likelihoods on measurement counts. My finance code touches none of that.
- "Solving interference" isn't even a coherent goal — interference is the *resource*
  quantum algorithms engineer, not a defect. The unwanted effects people conflate
  with it (decoherence, crosstalk) are hardware / error-correction problems, not
  statistics problems.
- The finance-specific machinery (return-based deflation, combinatorial CV) actively
  does **not** transfer, because quantum data has a *known* Born-rule null and
  i.i.d. resampleable shots.

So repurposing the finance tool would have been dressing up statistics as physics.
I didn't want to build that.

## The right question

What *does* transfer is not code but an **ethos**: correct for multiple looks,
distrust in-sample fit, hold out data, report honest p-values, and default to *deny*.
The quantum-experiment literature has a documented weakness here — a 2026 review of
81 quantum-error-mitigation papers found only ~25% used any inferential statistics,
and high-profile results have turned on naive 2σ error bars.

That reframed the project into something honest and new:

> **A framework-agnostic, finite-sample statistical referee for quantum measurement
> claims** — not a physics engine, a statistics layer that sits on top of one.

## The one thing I insisted on proving

Before building anything I checked the central claim with real numbers: is the
common "CHSH value exceeds 2 by *k* sigma" analysis actually *wrong*, or just loose?

It's wrong. In the CHSH *game* picture the local-realism null wins each round with
probability ≤ 3/4 regardless of history, so the honest p-value is a binomial tail
`P[Bin(n, 3/4) ≥ wins]` (valid even under the memory loophole). Simulating
experiments under that null shows the naive observed-SE test has a **false-positive
rate of ~7.5% against a nominal 5% at n = 80** — it *manufactures* certifications —
while the rigorous game tail sits at ~4%. At n = 8000 the gap vanishes. So the tool's
real, honest value is the **low-shot / multiple-comparison** regime, and the README
says exactly that rather than overselling.

That simulation is not a slide; it ships as `minos.selftest` and runs in CI. A
referee that can't referee itself has no business refereeing anyone else.

## Scope discipline (the hardest, most important design choice)

v1 does only what I could make **provably correct** and coverage-validate: linear
functionals (per-outcome probabilities, basis-state fidelity), memory-robust CHSH
p-values, multiple-comparison correction, and the self-test. Deliberately **out of
scope**, and documented rather than approximated: full density-matrix confidence
*regions* over the PSD cone, purity, general entanglement witnesses,
superposition-target fidelity (tomography/DFE), and SPAM modelling. Those need real
quantum-information depth; shipping a shaky number there would defeat the entire point.

## What the review found

I ran an adversarial, multi-dimension code review against the finished package. The
result I cared about most: **no mathematical bugs** — every formula (Wilson,
Clopper–Pearson, the game tail, Azuma, Holm/Benjamini–Hochberg) was independently
re-derived and matched `scipy`/`statsmodels`. The real findings were narrow paths
that violated the tool's own "never a silent pass" promise (an out-of-range α could
auto-certify; an invalid p-value could slip through `correction="none"`; the Qiskit
adapter could silently drop data). Those are now closed, with tests.

## What I'd do next, and what I won't claim

As a *business* this is a shelve-it: the people who most need it build it in-house or
won't pay, and the incumbents are free and lab-funded. As an engineering artifact it
stands on its own. The one thing that would justify taking it further is a working
quantum-information collaborator to correctness-check the harder estimators I left out
of v1 — a solo statistician shouldn't self-certify PSD-cone confidence regions.

I'd rather ship a small, correct, honest tool that knows its limits than a big one
that pretends. That judgement — knowing which idea to kill and which to keep — is the
part of this project I'm most proud of.
