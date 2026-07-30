# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

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
