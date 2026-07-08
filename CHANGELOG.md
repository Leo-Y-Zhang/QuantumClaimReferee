# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

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
