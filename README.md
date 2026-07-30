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
minos selftest --adversarial   # referee history-dependent memory-LHV adversaries
minos demo                     # the scarce-vs-plentiful worked example + a best-of-6 scan
minos plan --S 2.4 --alpha 0.05 --power 0.9   # exact power analysis: 604 rounds needed
```

At `n = 80` the naive observed-SE test **certifies** data the rigorous game tail calls
**UNDERPOWERED**, and its false-positive rate exceeds α. At `n = 8000` all methods agree.
The tool earns its keep in the **low-shot / multiple-comparison** regime — and says so.

The scarce-data verdict, verbatim (66 wins in 80 rounds). The naive p-value clears
0.05 and would certify; the referee refuses, and prices the fix:

```text
CHSH: S = 2.600  2.6000  [1.8194, 3.1423]  (95% wilson->S, n=80)
  status              : UNDERPOWERED
  p (memory-robust)   : 7.399e-02   <- use this
  p (Azuma, loose)    : 4.066e-01
  p (naive, observed) : 3.874e-02   (for contrast only)
  assumptions         : settings_randomised_per_round, no_signaling
  rounds for power    : ~255 (vs 80 run) for 90% power at alpha=0.05, IF the observed win rate 0.8250 persists
```

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

```text
$ minos plan --S 2.4 --alpha 0.05 --power 0.9
PLAN: 604 rounds  (hypothesis: win rate 0.8000, S = 2.400)
  certify iff wins >=  : 471  (alpha=0.05)
  exact power          : 0.9007  (target 0.9)
  note                 : minimal n meeting the target; exact binomial
                         power is sawtoothed, so a larger n can dip
                         below the target again
```

The non-obvious part: exact binomial power is **non-monotone in `n`** (a sawtooth —
each time the integer critical win count steps up, the power momentarily drops), so
a bisection over `n` is invalid. `minos` scans and returns the first `n` meeting the
target; by the same sawtooth, some larger `n` can dip below the target again, and the
output says so. An `UNDERPOWERED` verdict now also carries a planning hint: roughly
how many rounds 90% power would take **if** the observed win rate persists — labelled
as the assumption it is, since the observed rate is an estimate, not the truth.

## The memory loophole, attacked for real: `minos selftest --adversarial`

Claiming memory-robustness is cheap; `minos` attacks itself to earn it. The
adversarial self-test plays full sequential CHSH games against history-dependent
local players — each round the adversary picks one of the 16 deterministic local
strategies as *any* function of all past settings, past outcomes, and private
randomness; the referee then draws settings from a private RNG stream the
adversary never sees. The battery ships four adversaries: a memoryless
bound-saturator, a greedy-denominator attack on the naive per-setting estimator,
win-stay/lose-shift (outcome memory), and quit-while-ahead (score memory).

The headline measurement (seed 0, 4000 games per point): the greedy-denominator
adversary — a genuinely local player — drives the field-habit per-setting sigma
test to a **0.2953 false-positive rate at n = 80** against a nominal 0.05, and
more data does not fix it (0.2873 at n = 320, 0.2555 at n = 1000, 0.2675 at
n = 4000). The memory-robust game tail stays within Monte-Carlo noise of the
exact ceiling `P[Bin(n, 3/4) ≥ c_α(n)]` against every adversary — the same
exact-binomial quantity `minos plan` is built on. Excerpt at `n = 80`:

```text
$ minos selftest --adversarial --n 80
Adversarial memory-loophole self-test: n=80 rounds, 2000 runs per adversary (alpha=0.05)
  exact ceiling for ANY memory-LHV adversary: P[Bin(80, 3/4) >= 67] = 0.0421

greedy_denominator:
  certified (all false) : 0.0470  vs exact ceiling 0.0421 (MC sigma 0.0045)  <- bounded
  naive per-setting     : 0.2940  <- INVALID (inflated)
  verdicts              : UNDERPOWERED 0.4190, NOT_CERTIFIED 0.5340
  mean win rate         : 0.7511  (classical bound 0.7500)
```

The non-obvious part: memory moves *per-setting* statistics, never the pooled
win count — any history-measurable choice of sacrificed setting keeps the
conditional win probability at exactly 3/4, so the suite checks that
sacrifice-class adversaries *match* `Binomial(n, 3/4)` in pooled wins, not
merely stay below it. That is precisely why `minos` certifies from pooled wins.
Optional stopping (choosing `n` adaptively) is a different loophole and is
documented as out of scope.

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
| Adversarial validation | history-dependent memory-LHV adversaries (Barrett et al., PRA 66 042111; Gill quant-ph/0301059) refereed against the exact ceiling `P[Bin(n, 3/4) ≥ c_α]` (in `minos.adversary`) |

## Tests

```bash
python -m pytest        # 137 tests
```

The suite covers CHSH certification and its guardrails, both interval methods, the
multiple-comparison corrections, the default-deny verdict logic, CLI exit codes,
the Monte-Carlo coverage self-test itself, the power analysis (hand-computed
exact binomial cases, agreement with the shipped verdict on every win count, a
seeded Monte-Carlo certification-rate cross-check, and the sawtooth regression),
and the adversarial battery (the derived strategy table, the exact-binomial
ceiling against every adversary, sacrifice-class pooled wins matching
`Binomial(n, 3/4)`, rejection of non-local strategy indices, and a regression
proving an adversary that clones its RNG to peek at upcoming settings gains
nothing).
Requires Python 3.11+; CI runs on Python 3.13.

## License

Proprietary - All Rights Reserved (c) 2026 GreenPandaTech - portfolio viewing only.
