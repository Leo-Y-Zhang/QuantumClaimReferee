# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-07-31

### Added
- Adversarial memory-loophole self-test (`minos.adversary`, `minos selftest
  --adversarial`): the referee now *demonstrates*, not just cites, that its
  game-tail certification bounds history-dependent local players. The model is
  the fully general deterministic-per-round memory-LHV adversary: each round it
  commits one of the 16 deterministic local strategy pairs as any function of
  all past settings, past outcomes, and private randomness; the referee then
  draws settings from a private RNG stream the adversary never sees. Both
  constraints are structural: strategy indices are range-checked every round,
  and a regression test proves an adversary that clones its own generator to
  peek ahead stays at the classical bound.
- The shipped battery: a memoryless bound-saturator reference, a
  greedy-denominator adversary (one-step-optimal attack on the naive
  per-setting correlator estimator, derivation in its docstring), a
  win-stay/lose-shift outcome-memory adversary, and a quit-while-ahead
  score-memory adversary. `chsh_adversarial_false_positive_rates` referees the
  battery with the shipped verdict machinery (`critical_wins`, spot-checked
  against `chsh()` itself -- drift raises) and returns per-adversary
  scorecards against the exact ceiling `P[Bin(n, 3/4) >= c_alpha(n)]`.
- `naive_persetting_pvalues`: the field-habit per-setting sigma test, included
  FOR CONTRAST ONLY as the thing the adversaries attack. Measured (seed 0,
  4000 trials per point): the greedy-denominator adversary inflates its
  false-positive rate to 0.2953 at n=80 against a nominal 0.05, and the
  inflation does not vanish with more data (0.2873 at n=320, 0.2555 at
  n=1000, 0.2675 at n=4000), while every adversary's memory-robust
  certification rate stays within Monte-Carlo noise of the exact binomial
  ceiling. Sacrifice-class adversaries are checked to MATCH Binomial(n, 3/4)
  in pooled wins, not merely stay below it: memory moves per-setting
  statistics, never the pooled win count, which is why minos certifies from
  pooled wins.
- Optional stopping (choosing `n` adaptively) is a different loophole and is
  documented as out of scope for the simulator.

### Fixed
- Impossible per-setting tallies (wins above counts, negative wins, non-finite
  entries) passed to `naive_persetting_pvalues` now raise instead of silently
  yielding a confident p-value from unphysical data.
- `chsh_adversarial_false_positive_rates` rejects duplicate adversary names up
  front instead of silently overwriting one scorecard.

## [0.2.0] - 2026-07-30

### Added
- Exact finite-sample power analysis (`minos.power`, `minos plan`): the minimal
  number of rounds `n` such that the shipped CHSH certification succeeds at
  `alpha` with at least the target probability, for a hypothesised win rate or
  CHSH `S` value. All computations are exact Binomial (no Gaussian shortcut),
  and the critical win count is derived from `game_tail_pvalue` -- the same
  function the verdict logic decides on -- so plan and verdict cannot disagree.
- The scan handles the sawtooth: exact binomial power is non-monotone in `n`
  (the power dips each time the integer critical count steps up), so bisection
  is invalid; a linear scan returns the true minimal `n`, with a regression
  test around a documented dip (win rate 0.9, alpha 0.05: minimal n = 55, but
  power at n = 56 falls back below the 0.9 target).
- `UNDERPOWERED` CHSH summaries now state approximately how many rounds would
  reach 90% power at the observed win rate, labelled with the assumption that
  the observed rate persists.
- Public `game_tail_pvalue`, `omega_to_s`, `s_to_omega` helpers on `minos.chsh`.

## [0.1.0] - 2026-07-01

First release. A finite-sample statistical referee for quantum measurement claims.

### Added
- Framework-agnostic measurement-counts ingestion (`CountsDataset`, with a
  Qiskit-style adapter) that default-denies when the POVM is undeclared.
- Binomial confidence intervals for linear functionals: Wilson score and
  Clopper-Pearson exact.
- Fidelity-to-a-basis-state estimation as an honest linear functional.
- Memory-robust CHSH certification: the game-tail p-value
  `P[Bin(n, 3/4) >= wins]` (Gill / Bierhorst), a conservative Azuma bound, and a
  naive-Gaussian contrast reported for comparison only.
- Multiple-comparison corrections: Holm, Benjamini-Hochberg, Bonferroni.
- Default-deny `Study` / `Verdict` and a deterministic `referee_report`.
- Monte-Carlo coverage self-test (`minos.selftest`) demonstrating that the
  naive CHSH test is miscalibrated at low shot counts while the game tail is not.
- `minos` command-line interface with CI-gate exit codes.

### Scope
- v1 is restricted to linear functionals and CHSH. Full-state confidence regions,
  purity, general entanglement witnesses, and SPAM modelling are intentionally out
  of scope and documented as such.
