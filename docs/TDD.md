# Quantum Claim Referee — technical design

Derived from the code at `src/qcref/`, not from the README.
Requirements: [PRD.md](PRD.md).

## One criterion, enforced structurally

A pure-Python library with a thin CLI over it. No server, no database, no
persistence, no network: every entry point is a function from integers to a
frozen dataclass, and the CLI is an argparse shim that prints `.summary()` and
returns an exit code.

The design problem was never distribution. It was making sure the *same*
certification criterion is used everywhere, so the verdict, the power analysis,
the self-test and the adversarial harness cannot drift apart and quietly disagree
about what "certified" means.

That is enforced by structure rather than convention. `chsh.game_tail_pvalue` is
the single definition of the criterion. `power.critical_wins` inverts *that
function* — seeding from `binom.isf`, then walking to the exact boundary using
the shipped p-value — rather than deriving a parallel threshold. `selftest`
classifies simulated runs with `critical_wins` and then re-runs a 64-trial
subsample through `chsh()` itself, raising if the two ever disagree. And the
verdict vocabulary lives in one 15-line module (`status.py`), so no component can
invent a fifth status string.

## Module graph

Dependencies point downward; there are no cycles at import time.

```
status.py        verdict vocabulary: CERTIFIED / NOT_CERTIFIED / UNDERPOWERED / ASSUMPTIONS_UNMET
_version.py      single source of the version string
   |
intervals.py     Wilson + Clopper-Pearson, returning a self-describing Interval
counts.py        CountsDataset: {setting: {bitstring: shots}} + declared POVM
multiple.py      Holm / Benjamini-Hochberg / Bonferroni adjusted p-values
   |
chsh.py          game_tail_pvalue + chsh() -> CHSHResult          (uses intervals, status)
fidelity.py      linear functionals over a CountsDataset          (uses counts, intervals)
verdict.py       Study -> Verdict, default-deny over a bundle     (uses multiple, status)
   |
power.py         critical_wins / certification_power / plan_rounds (uses chsh)
adversary.py     memory-LHV players + the game referee             (numpy only)
   |
selftest.py      coverage + adversarial harnesses                  (uses chsh, power, adversary)
report.py        deterministic referee report                      (uses verdict, _version)
cli.py           argparse shim + exit-code contract                (uses the above)
```

One deliberate exception: `CHSHResult._plan_hint` imports `power.plan_rounds`
*inside the method*, because `power` imports `chsh` at module level. The local
import is the cycle break and is commented as such.

## The integrity boundary that was breached once

There is no access control anywhere — no server, no database, no RLS, no
security-definer function, no `anon` grant, no auth, no persisted state. Nothing
is granted, so nothing can be revoked. (Recorded rather than deleted, so anyone
adding a hosted surface later is confronted with the fact that it would need
one.)

The one integrity boundary in the codebase is *internal*, and it is worth
documenting because it has been breached. In `adversary.play_chsh_game`, the
referee's score ledger and the referee's RNG are private to the referee.

**The adversary sees only a frozen history.** It chooses its strategy for round
*i* from a `GameHistory` whose arrays are read-only copies that own their memory
— `_sealed` does `.copy()` then `setflags(write=False)`, so `.base is None` and
no reference path leads back to the ledger. An earlier draft passed the live
tally arrays, and an adversary that wrote `history.wins += 10` recorded 869 wins
out of 80 rounds with no error. Both write paths now raise, with regression
tests.

**The settings RNG is a separate stream.** Settings are drawn from `referee_rng`,
spawned independently from the same `SeedSequence` as the adversary's, so cloning
or exhausting the adversary's own generator reveals nothing about the settings to
come. An earlier draft shared one generator, and a subclass that deep-copied it
could win every round while passing the range checks. Regression test in
`test_adversary.py`.

**Strategy indices are range-checked to `0..15` every round**, so every play is
one of the 16 deterministic local strategies.

Those three checks are what make "we attacked our own bound" a claim rather than
a slogan: a passing run is LHV-with-independent-settings *by construction*.

## Frozen dataclasses

No database and no migration. The "data model" is the frozen dataclasses that
carry state between modules. All are `@dataclass(frozen=True)`, and all validate
at construction.

| Type | Fields that matter | Invariant enforced where |
|---|---|---|
| `Interval` | `point, lo, hi, level, method, n, assumptions` | `_validate` rejects `n <= 0`, `k` outside `[0, n]`, `level` outside `(0, 1)` |
| `CountsDataset` | `settings: {label: {bitstring: shots}}`, `povm: str \| None`, `meta` | `__post_init__`: non-empty, equal-length 0/1 bitstrings, no negative counts, no zero-shot setting |
| `CHSHResult` | `wins, rounds, omega, S, S_ci, p_memory_robust, p_azuma, p_naive_observed, alpha, status, assumptions, unmet_reason` | `chsh()` validates before constructing; `unmet_reason` is non-empty only for the Tsirelson refusal |
| `Hypothesis` / `HypothesisResult` | `name, pvalue, estimate, assumptions_met, detail` | `Study.add_hypothesis` rejects a non-finite or out-of-`[0,1]` p-value on any *testable* hypothesis |
| `Verdict` | `classification, alpha, correction, results` | built only by `Study.run()` |
| `PlanResult` | `rounds, critical_wins, power, target_power, win_rate, S, alpha` | built only by `plan_rounds`, which has already met the target |
| `GameHistory` | `rounds_played, setting_counts, setting_wins, wins, last_setting, last_won` | frozen, and every array a **read-only copy that owns its memory** (`_sealed`) |
| `AdversaryRuns`, `AdversarialReport` | per-adversary tallies and rates | ledger integrity checked after the last round |

`povm` is the one nullable field that matters. `None` means *undeclared*, and
`require_povm()` raises rather than assuming ideal projective measurement. That
default is the difference between an honest refusal and a biased estimate.

## Public API, and six contracts

Re-exported from `qcref/__init__.py`, where `__all__` is explicit.

```python
chsh(wins, rounds, *, level=0.95, alpha=0.05,
     setting_randomness_declared=False, no_signaling=True) -> CHSHResult
game_tail_pvalue(wins, rounds) -> float          # P[Bin(rounds, 3/4) >= wins]
omega_to_s(omega) -> float                       # S = 8*omega - 4
s_to_omega(s) -> float
wins_from_setting_counts({(x,y): {(a,b): shots}}) -> (wins, rounds)

CountsDataset(settings, povm=None, meta={})      # .from_counts, .from_qiskit
probability_interval(dataset, bitstring, *, setting, method, level) -> Interval
fidelity_to_basis_state(dataset, bitstring, ...) -> Interval

wilson_interval(k, n, level=0.95) -> Interval
clopper_pearson_interval(k, n, level=0.95) -> Interval
holm(pvals) / benjamini_hochberg(pvals) / bonferroni(pvals) -> list[float]

Study(alpha=0.05, correction="holm").add(name, pvalue, *, estimate="",
      assumptions_met=True, detail="").run() -> Verdict
referee_report(verdict, *, title, caveats, meta=None) -> str

critical_wins(rounds, alpha) -> int
certification_power(rounds, win_rate, *, alpha=0.05) -> float
plan_rounds(win_rate, *, alpha=0.05, power=0.9, max_rounds=1_000_000) -> PlanResult
```

These six are the ones a caller gets wrong.

`chsh` certifies iff `game_tail_pvalue(wins, rounds) <= alpha` — and only after
passing two gates first: `setting_randomness_declared` must be `True`, and `S`
must not exceed the Tsirelson bound. Both failures return `ASSUMPTIONS_UNMET`.

`level` only affects the reported interval on *S*. It never affects the status.

`wins_from_setting_counts` requires all four input pairs and raises otherwise.
Its docstring carries an explicit warning that pooling fixed per-setting *blocks*
discards round order and breaks the martingale premise, so the returned tuple is
a valid memory-robust input only if settings were drawn per round.

`plan_rounds` refuses `win_rate <= 3/4`. At or below the classical bound the
certification is calibrated, so its success probability never exceeds α and no
target power is reachable at any *n*.

`Study` is default-deny: `CERTIFIED` only if *every* hypothesis clears α after
correction, and any untestable hypothesis makes the whole verdict
`ASSUMPTIONS_UNMET`.

`referee_report` is deterministic — no timestamps, no randomness — so identical
inputs render byte-identical (`test_report_is_deterministic`).

## Exit codes

`qcref <chsh|plan|selftest|demo>`; the flows are in [App Flow](APP_FLOW.md).

| Code | Means |
|---|---|
| 0 | `chsh` returned CERTIFIED; or `plan` / `selftest` / `demo` completed |
| 1 | `chsh` returned anything else (NOT_CERTIFIED, UNDERPOWERED, ASSUMPTIONS_UNMET) |
| 2 | argparse usage error — including every `ValueError` raised by the library, which `main()` catches and routes to `parser.error` so a bad input is a usage message, not a traceback |

## The ways a verdict could be wrong

| What breaks | Who notices | How we detect it | How we undo it |
|---|---|---|---|
| A caller pools an incomplete CHSH run (one setting pair missing) | The caller, immediately | `wins_from_setting_counts` raises: the game needs all four pairs | N/A — refused before a number exists. Before the guard, a classical device answering `a=b=0` on `(0,0)` alone gave ω=1, S=4.0, *p*=0: a certified classical device |
| Data imply *S* above the Tsirelson bound | The caller | `chsh` returns `ASSUMPTIONS_UNMET` with `unmet_reason` naming the likely cause (a missing or non-uniform setting) | N/A — refused, not certified |
| POVM undeclared | The caller | `require_povm()` raises on every estimation path | N/A — refused |
| Settings not randomised per round | Nobody, if the caller lies | `setting_randomness_declared` defaults to `False` → `ASSUMPTIONS_UNMET`. **This is an unverifiable declaration and is the tool's largest residual risk**; it is stated in the assumptions line of every result and in the report caveats |
| An invalid p-value reaches a `Study` | The caller | Validated in `add_hypothesis` *and* re-checked in `run()`, because `correction="none"` does not re-validate. Before that, a negative p passed `adj_p <= alpha` and falsely CERTIFIED | N/A — raises |
| Impossible per-setting tallies (`wins > counts`, fractional, negative) | The caller | `naive_persetting_pvalues` raises; previously a negative variance became a NaN SE that the degenerate branch mapped to a confident *p* = 0 | N/A — raises |
| Self-test drifts from the shipped verdict | CI | `_spot_check_against_verdict` re-runs 64 trials of every batch through `chsh()` and raises `RuntimeError` on any disagreement | Fix the drift; the two must agree by construction |
| Adversary harness leaks the live ledger in a refactor | CI | Post-game integrity check on the tally invariants (`wins <= rounds`, per-setting sums match) raises `RuntimeError` | Revert the refactor |
| Qiskit result holds several experiments | The caller, if warnings are visible | `UserWarning`, then **only the first experiment is used**. This is the one place the package proceeds after a surprise instead of refusing; it is a knowingly accepted convenience |
| A future n makes the power scan slow | The caller | `plan_rounds` scans linearly to `max_rounds` (default 1e6) and raises with a "raise max_rounds" message rather than hanging silently |
| CI hangs | The owner | 15-minute `timeout-minutes` on both jobs |

## Rollback, and the failure that cannot be rolled back

There is no state, so reverting is trivial and complete: `git revert`, or pin the
previous version and reinstall. Nothing to un-migrate, no data to restore, no
deploy to drain. The nearest thing to a schema is the public API, versioned
semantically in `_version.__version__` (0.3.0 today) and reported in every
rendered report. The Minos → Quantum Claim Referee rename is the one breaking
change to date — the import name and console command moved from `minos` to
`qcref`, and the distribution from `minos` to `quantum-claim-referee`. No
behaviour changed, the full suite passed unchanged, and callers need
`pip uninstall minos && pip install -e .`.

The failure that *cannot* be rolled back is a wrong number already cited in a
paper. That risk is addressed at the artefact level rather than the deploy level:
every report carries the package version, the numpy and scipy versions, and a
sha256 of the hypothesis inputs, so a reader can tell exactly which code produced
a number and a corrected report is distinguishable from the original. If a
mathematical defect were found, the response is a new version plus a note in
`CHANGELOG.md` naming the affected versions — the precedent already set there for
the ledger and `naive_persetting_pvalues` fixes.

## What the 145 tests are there to falsify

145 tests, `pytest -q`, plus `ruff check src tests`; both gate CI on Python 3.13.

**Positive.** `chsh(6400, 8000, randomised)` certifies. `plan --S 2.4` returns 604
rounds with critical count 471 and exact power 0.9007. The report contains the
headline, the version and the input hash. Every interval method covers at or
above its nominal level.

**Negative.** The naive observed-SE test exceeds α under the null while the game
tail does not. Every memory adversary's certification rate stays within
Monte-Carlo noise of the exact ceiling `P[Bin(n, 3/4) >= c_alpha(n)]`. An
adversary that clones its RNG to peek at coming settings gains nothing; one that
writes the score ledger raises instead of certifying. A negative p-value cannot
certify through `correction="none"`. A non-local strategy index is rejected. An
undeclared POVM refuses.

**Boundary.** The sawtooth regression — at win rate 0.9 and α 0.05 the minimal
*n* is 55 by arithmetic, but power at *n* = 56 falls back below target and the
physical floor pushes the shipped answer to 60. `critical_wins` agreeing with the
shipped verdict on *every* win count. All-zero histograms. Scalar input to a
correction. `level` at 0 and 1. And sacrifice-class adversaries **matching**
`Binomial(n, 3/4)` in pooled wins rather than merely staying below it.

That last one is the subtle case, and it is the reason the tool certifies from
pooled wins: memory moves *per-setting* statistics, never the pooled win count.

## The order the commands arrived in

As built: `status` and `intervals` → `counts` and `fidelity` → `chsh`, the
flagship → `multiple` and `verdict` → `report` → `selftest` (0.1.0) → `power` and
`plan` (0.2.0) → `adversary` and the adversarial self-test (0.3.0) → the
hardening pass that closed the default-deny holes found in review.

## Loose ends in the reporting

Should the reproducibility hash cover the policy (α, correction) and the
estimates, not just `(name, raw_p, status)`? Today two reports over identical
hypotheses with different corrections carry the same `inputs sha256` whenever no
status flips.

Should the reported interval on *S* be truncated at the Tsirelson bound? The code
refuses a *point* estimate above it as physically impossible, but prints an
interval whose upper end can exceed it — `[1.8194, 3.1423]` at *n* = 80, for
example. Untruncated is the statistically conventional choice, and it is arguably
inconsistent with the guard.
