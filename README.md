# Minos - an honest, finite-sample statistical referee for quantum measurement claims

[![CI](https://github.com/GreenPandaTech/Minos/actions/workflows/ci.yml/badge.svg)](https://github.com/GreenPandaTech/Minos/actions/workflows/ci.yml)

Point `minos` at raw measurement counts and get a *default-deny* verdict on whether
a fidelity or entanglement/CHSH claim actually survives finite-sample scrutiny — with a
reproducible referee's report you can attach to a paper or an internal review. The
non-obvious part: the field's usual "*S* beats 2 by *k* sigma" test is *miscalibrated*
at realistic shot counts (it certifies violations that are not there), so `minos` uses a
finite-sample game bound that stays valid even under the memory loophole — and ships a
coverage self-test that proves the gap.

> **A measurement referee, not a device.** It does not run circuits, mitigate error, or
> improve results. It tells you how much to trust a claim. Nothing here is a physics
> engine — it is the statistics layer that sits on top of one.

## Why it exists

Mainstream quantum SDKs already give *piecemeal* error bars (Qiskit bootstrap SEs,
pyGSTi likelihood-ratio regions, PennyLane shot variance). What is missing — and what
this provides — is:

1. **Valid finite-sample CHSH / Bell p-values** that replace the field's common
   "S exceeds 2 by *k* sigma" habit. That habit is not merely loose — it is
   *miscalibrated*: it certifies violations that are not there. minos uses the
   game-based bound that is valid even under the **memory loophole**
   (Gill / Bierhorst / Elkouss–Wehner).
2. A **meta-statistical referee discipline** — multiple-comparison correction,
   a default-deny verdict, and a deterministic report — absent from quantum tooling.
3. **One counts-in schema** across Qiskit / Cirq / PennyLane (dicts of `bitstring→shots`).
4. A **coverage self-test** shipped *in the box*, because a referee must referee itself.

## The wedge, in one run

```bash
minos selftest --n 80          # naive CHSH false-positive rate ~0.074 vs nominal 0.05
minos selftest --n 8000        # at high statistics the gap vanishes
minos demo                     # the scarce-vs-plentiful worked example + a best-of-6 scan
minos plan --S 2.4 --alpha 0.05 --power 0.9   # exact power analysis: 604 rounds needed
```

At `n = 80` the naive observed-SE test **certifies** data the rigorous game tail calls
**UNDERPOWERED**, and its false-positive rate exceeds α. At `n = 8000` all methods agree.
The tool earns its keep in the **low-shot / multiple-comparison** regime — and says so.

## Install

```bash
python -m pip install -e .
```

Dependencies are limited to `numpy` and `scipy`. No network access is ever performed.

## Library usage

```python
import minos as qr

# CHSH from game wins/rounds (settings must be declared randomised, or it default-denies)
r = qr.chsh(6400, 8000, setting_randomness_declared=True)
print(r.summary())          # -> status CERTIFIED, with the memory-robust p-value

# Fidelity to a computational-basis target = a linear functional with an honest CI
ds = qr.CountsDataset.from_counts({"000": 940, "111": 40, "001": 20}, povm="ideal_projective")
print(qr.fidelity_to_basis_state(ds, "000"))     # 0.9400  [0.9235, 0.9531]  (95% wilson, n=1000)

# Bundle hypotheses, correct for the search, get a default-deny verdict + report
study = qr.Study(alpha=0.05, correction="holm")
study.add("ghz_fidelity_gt_0.9", 6e-4, estimate="F=0.947")
study.add("chsh_pair_23", r.p_memory_robust, estimate=f"S={r.S:.2f}")
print(qr.referee_report(study.run(), title="device_run_1"))

# Plan the next run: minimal rounds so certification succeeds with 90% probability
plan = qr.plan_rounds(qr.s_to_omega(2.4), alpha=0.05, power=0.9)
print(plan.summary())        # -> 604 rounds, certify iff wins >= 471, exact power 0.9007
```

## Planning an experiment: `minos plan`

Before running rounds, ask how many you need: `minos plan --S 2.4 --alpha 0.05
--power 0.9` (or `--win-rate 0.8`) returns the **minimal** `n` such that the shipped
certification — the same game-tail criterion `chsh` decides on, reused, not
reimplemented — succeeds with probability ≥ the target when the per-round win rate
truly is the hypothesised one. Everything is exact Binomial; no Gaussian
approximation is used anywhere.

The non-obvious part: exact binomial power is **non-monotone in `n`** (a sawtooth —
each time the integer critical win count steps up, the power momentarily drops), so
a bisection over `n` is invalid. `minos` scans and returns the first `n` meeting the
target; by the same sawtooth, some larger `n` can dip below the target again, and the
output says so. An `UNDERPOWERED` verdict now also carries a planning hint: roughly
how many rounds 90% power would take **if** the observed win rate persists — labelled
as the assumption it is, since the observed rate is an estimate, not the truth.

## Verdict taxonomy (default-deny)

| Status | Meaning |
|---|---|
| `CERTIFIED` | Survives α *after* multiple-comparison correction |
| `NOT_CERTIFIED` | Fails the threshold, or a decisive non-violation (evidence *against*) |
| `UNDERPOWERED` | *(CHSH only)* on the violating side but the evidence does not clear α — the summary states roughly how many rounds 90% power would take at the observed win rate (if it persists) |
| `ASSUMPTIONS_UNMET` | POVM / setting-randomness undeclared → refused, not guessed |

The `Study` verdict emits `CERTIFIED` / `NOT_CERTIFIED` / `ASSUMPTIONS_UNMET`; the
CHSH certifier additionally uses `UNDERPOWERED` to distinguish "not enough shots" from
"evidence against a violation".

## Scope — and what is deliberately **out** of v1

v1 is restricted to what can be made **provably correct** and is coverage-validated:
linear functionals (per-outcome probabilities, basis-state fidelity), memory-robust
CHSH p-values, multiple-comparison correction, and the coverage self-test.

Out of scope for v1, and documented rather than approximated (issuing a shaky number
would defeat the point): full-density-matrix confidence *regions* over the PSD cone,
purity/Rényi entropy, general entanglement witnesses and negativity, superposition-target
fidelity (DFE/tomography), and SPAM/readout modelling. These need genuine
quantum-information depth and are the honest boundary of a statistics-first tool.

## Methods & citations

| Capability | Method / source |
|---|---|
| Proportion / fidelity CIs | Wilson score; Clopper–Pearson exact |
| Memory-robust CHSH p-values | game tail `P[Bin(n, 3/4) ≥ wins]` (Gill martingale / stochastic dominance; Bierhorst). Elkouss–Wehner (npj QI 2016) is a tighter near-optimal refinement, *not implemented* |
| Conservative CHSH bound | Azuma–Hoeffding |
| Multiple comparisons | Holm (1979); Benjamini–Hochberg (1995); Bonferroni |
| Power analysis (`minos plan`) | exact Binomial power against the same game-tail critical count the verdict uses; sawtooth-aware minimal-`n` scan (no Gaussian shortcut) |
| Coverage validation | Monte-Carlo empirical-coverage harness (in `minos.selftest`) |

## Tests

```bash
python -m pytest        # 107 tests
```

The suite covers CHSH certification and its guardrails, both interval methods, the
multiple-comparison corrections, the default-deny verdict logic, CLI exit codes,
the Monte-Carlo coverage self-test itself, and the power analysis (hand-computed
exact binomial cases, agreement with the shipped verdict on every win count, a
seeded Monte-Carlo certification-rate cross-check, and the sawtooth regression).
CI runs on Python 3.11-3.13.

## License

Proprietary - All Rights Reserved (c) 2026 GreenPandaTech - portfolio viewing only.
