"""History-dependent local-hidden-variable adversaries for the CHSH game.

The *memory loophole* (Barrett, Collins, Hardy, Kent, Popescu, PRA 66 042111, 2002):
in a real Bell test the rounds are sequential, and a classical device can condition
round ``i`` on everything that happened in rounds ``1 .. i-1`` -- past settings and
past outcomes are public after each round. Analyses that assume i.i.d. rounds are
therefore attackable in principle. This module implements such adversaries so that
:mod:`qcref.selftest` can *demonstrate*, not just cite, that the shipped game-tail
certification bounds every one of them.

The model is the fully general deterministic-per-round memory-LHV adversary:

* each round the adversary commits one of the 16 deterministic local strategy pairs
  ``(a: x -> a(x), b: y -> b(y))``, chosen as an arbitrary function of the public
  history and private randomness;
* the referee then draws the setting pair ``(x, y)`` uniformly at random,
  independent of everything the adversary knows;
* the round is won iff ``a(x) XOR b(y) == x AND y``.

Randomised (mixed) strategies are covered because the choice may use the supplied
RNG. Three structural enforcement points make the model honest, and all matter:

* *Locality per round*: strategy indices are range-checked every round, so every
  play is one of the 16 deterministic local strategies. The win table is derived
  from the game rule at import time, and no deterministic local strategy wins
  more than 3 of the 4 equally likely setting pairs (they win exactly 1 or 3 --
  the four CHSH targets XOR to 1, so no strategy can satisfy an even number of
  them).
* *Settings independence*: the referee draws settings from a **private**
  generator that is never passed to the adversary. The adversary's generator is
  spawned from the same seed but is statistically independent, so nothing
  reachable from :meth:`MemoryLHVAdversary.strategies` -- including cloning or
  exhausting its own RNG -- predicts the coming settings. (An earlier draft
  shared one generator between referee and adversary; a subclass that deep-copied
  it could read off the upcoming settings and win every round while passing the
  range checks. The regression test keeps that door shut.)
* *Ledger integrity*: the score is tallied in referee-private arrays the
  adversary never touches. Each round the adversary is handed a frozen
  :class:`GameHistory` snapshot whose arrays are read-only *copies*: in-place
  writes raise, field reassignment raises, and the copies own their memory
  (``.base is None``), so no reference path leads back to the referee's
  ledger. (An earlier draft passed the live tally arrays; an adversary that
  wrote ``history.wins += 10`` could then record physically impossible win
  counts -- in either direction -- without tripping any check. The regression
  tests keep that door shut too.)

Together these give every adversary representable here conditional win
probability at most ``3/4`` given any history, so the total win count is
stochastically dominated by ``Binomial(n, 3/4)`` and
:func:`qcref.chsh.game_tail_pvalue` remains a valid p-value against all of them
(Gill, quant-ph/0301059). The self-test checks that empirically rather than
trusting the theorem.

A sharper consequence worth knowing: an adversary whose only freedom is *which*
setting pair to sacrifice (playing a best 3-of-4 strategy every round) has
conditional win probability *exactly* ``3/4`` whatever it remembers, so its pooled
win count is exactly ``Binomial(n, 3/4)`` in distribution. Memory moves
*per-setting* statistics, never the pooled wins -- which is precisely why qcref
certifies from pooled wins. What memory *can* skew is the field's naive per-setting
correlator estimator ``S_hat = E00 + E01 + E10 - E11`` with its four random
denominators; see :func:`qcref.selftest.naive_persetting_pvalues` and
:class:`GreedyDenominatorAdversary`.

Scope note: adversaries here control outcomes given settings, not the number of
rounds. Optional stopping (choosing ``n`` adaptively) is a *different* loophole;
the game tail, like every fixed-``n`` p-value, presumes ``n`` was fixed in advance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SACRIFICE_TO_STRATEGY",
    "STRATEGY_WIN_TABLE",
    "AdversaryRuns",
    "FixedSacrificeAdversary",
    "GameHistory",
    "GreedyDenominatorAdversary",
    "MemoryLHVAdversary",
    "QuitWhileAheadAdversary",
    "WinStayLoseShiftAdversary",
    "default_adversaries",
    "play_chsh_game",
]

N_STRATEGIES = 16  # deterministic local strategy pairs: (a0, a1, b0, b1) bits
N_SETTINGS = 4  # setting pairs (x, y), encoded as s = 2*x + y


def _build_win_table() -> np.ndarray:
    """Derive the (16, 4) win table straight from the CHSH game rule."""
    table = np.zeros((N_STRATEGIES, N_SETTINGS), dtype=bool)
    for strat in range(N_STRATEGIES):
        a0, a1 = (strat >> 3) & 1, (strat >> 2) & 1
        b0, b1 = (strat >> 1) & 1, strat & 1
        for s in range(N_SETTINGS):
            x, y = s >> 1, s & 1
            a = a1 if x else a0
            b = b1 if y else b0
            table[strat, s] = (a ^ b) == (x & y)
    return table


STRATEGY_WIN_TABLE = _build_win_table()
"""``STRATEGY_WIN_TABLE[strategy, setting]`` is True iff that deterministic local
strategy wins the round on that setting pair. Row sums are all 1 or 3."""
STRATEGY_WIN_TABLE.setflags(write=False)


def _build_sacrifice_map() -> np.ndarray:
    """For each setting pair, a strategy that wins exactly the other three."""
    out = np.empty(N_SETTINGS, dtype=np.int64)
    for s in range(N_SETTINGS):
        wanted = np.ones(N_SETTINGS, dtype=bool)
        wanted[s] = False
        matches = np.nonzero((STRATEGY_WIN_TABLE == wanted).all(axis=1))[0]
        if matches.size == 0:  # pragma: no cover - impossible by the parity argument
            raise AssertionError(f"no strategy sacrifices exactly setting {s}")
        out[s] = matches[0]
    return out


SACRIFICE_TO_STRATEGY = _build_sacrifice_map()
"""``SACRIFICE_TO_STRATEGY[s]`` plays optimally except on setting pair ``s``."""
SACRIFICE_TO_STRATEGY.setflags(write=False)

# A strategy that wins only one setting pair: conditional win probability 1/4.
_MIN_WIN_STRATEGY = int(np.nonzero(STRATEGY_WIN_TABLE.sum(axis=1) == 1)[0][0])


@dataclass(frozen=True)
class GameHistory:
    """The public record an adversary may condition on, across parallel games.

    All arrays have leading dimension ``trials`` (independent games simulated in
    parallel); ``setting_*`` arrays have a trailing dimension of 4 setting pairs.

    Instances handed to :meth:`MemoryLHVAdversary.strategies` are immutable
    snapshots of the referee's private ledger: the dataclass is frozen and the
    arrays are read-only copies, so an adversary can look at the score but
    never write it.
    """

    rounds_played: int
    setting_counts: np.ndarray  # (trials, 4) rounds seen per setting pair
    setting_wins: np.ndarray  # (trials, 4) wins per setting pair
    wins: np.ndarray  # (trials,) total wins
    last_setting: np.ndarray  # (trials,) last drawn setting pair, -1 before round 1
    last_won: np.ndarray  # (trials,) whether the last round was won


class MemoryLHVAdversary:
    """Base class: a local-hidden-variable player with unlimited memory.

    Subclasses implement :meth:`strategies`, returning one deterministic local
    strategy index (``0..15``) per trial for the coming round. The choice may
    depend on the full :class:`GameHistory` and on ``rng`` (mixed strategies);
    it cannot depend on the coming round's settings, which are drawn afterwards
    from the referee's private generator -- ``rng`` here is the adversary's own
    stream, independent of the referee's, so inspecting or cloning it reveals
    nothing about the settings to come. The history is a frozen read-only
    snapshot: writing to it raises rather than corrupting the score.
    """

    name: str = "memory_lhv"

    def begin(self, trials: int, rng: np.random.Generator) -> None:
        """Reset per-run internal memory. Default: stateless."""

    def strategies(self, history: GameHistory, rng: np.random.Generator) -> np.ndarray:
        """Return the (trials,) array of strategy indices for the next round."""
        raise NotImplementedError


class FixedSacrificeAdversary(MemoryLHVAdversary):
    """Memoryless reference: always sacrifice the same setting pair.

    Plays a best 3-of-4 strategy every round, so its win count saturates the
    classical bound: exactly ``Binomial(n, 3/4)``. This is the baseline the
    history-dependent adversaries are compared against.
    """

    name = "memoryless_saturator"

    def __init__(self, pair: int = 3) -> None:
        if pair not in range(N_SETTINGS):
            raise ValueError("pair must be in {0, 1, 2, 3}")
        self._strategy = int(SACRIFICE_TO_STRATEGY[pair])

    def strategies(self, history: GameHistory, rng: np.random.Generator) -> np.ndarray:
        return np.full(history.wins.shape[0], self._strategy, dtype=np.int64)


class GreedyDenominatorAdversary(MemoryLHVAdversary):
    """Sacrifices the setting pair with the largest count so far (settings memory).

    This is the one-step-optimal greedy attack on the naive per-setting estimator
    ``S_hat = 2 * sum_xy omega_xy - 4``: sacrificing pair ``j`` changes the
    expected increment of ``sum omega_xy`` by ``-(1/4) / (N_j + 1)`` relative to
    winning there, so the greedy adversary sacrifices the pair with the largest
    current denominator ``N_j``, steering its losses into the settings where they
    dilute the most. Its pooled win count stays exactly ``Binomial(n, 3/4)``.
    """

    name = "greedy_denominator"

    def strategies(self, history: GameHistory, rng: np.random.Generator) -> np.ndarray:
        sacrifice = np.argmax(history.setting_counts, axis=1)
        return SACRIFICE_TO_STRATEGY[sacrifice]


class WinStayLoseShiftAdversary(MemoryLHVAdversary):
    """Keeps its sacrificed pair after a win, rotates it after a loss (outcome memory).

    A classic outcome-feedback rule: the sacrificed pair performs a deterministic
    walk driven by the loss sequence. Demonstrates that conditioning on past
    *outcomes* buys the adversary nothing against the pooled game tail.
    """

    name = "win_stay_lose_shift"

    def __init__(self) -> None:
        self._sacrifice: np.ndarray | None = None

    def begin(self, trials: int, rng: np.random.Generator) -> None:
        self._sacrifice = rng.integers(0, N_SETTINGS, size=trials)

    def strategies(self, history: GameHistory, rng: np.random.Generator) -> np.ndarray:
        if self._sacrifice is None:
            # A real error, not an assert: asserts are stripped under python -O,
            # and SACRIFICE_TO_STRATEGY[None] would then silently broadcast to
            # a wrong-shape (1, 4) array (numpy treats None as np.newaxis).
            raise RuntimeError("begin() must be called before strategies()")
        if history.rounds_played > 0:
            lost = ~history.last_won
            self._sacrifice = np.where(
                lost, (self._sacrifice + 1) % N_SETTINGS, self._sacrifice
            )
        return SACRIFICE_TO_STRATEGY[self._sacrifice]


class QuitWhileAheadAdversary(MemoryLHVAdversary):
    """Plays optimally until ahead of the 3/4 pace, then throws rounds to bank the lead.

    Whenever ``wins - (3/4) * rounds_played >= lead`` it switches to a strategy
    that wins only 1 of the 4 setting pairs (conditional win probability 1/4),
    trying to freeze a lucky streak. Its conditional win probability is genuinely
    history-dependent and *strictly* dominated by 3/4, so the game tail bounds it
    with room to spare -- and its overall win rate falls below the bound.
    """

    name = "quit_while_ahead"

    def __init__(self, lead: int = 4) -> None:
        if lead < 1:
            raise ValueError("lead must be a positive integer")
        self._lead = lead

    def strategies(self, history: GameHistory, rng: np.random.Generator) -> np.ndarray:
        ahead = 4 * history.wins - 3 * history.rounds_played >= 4 * self._lead
        sacrifice_all_but_one = np.full(history.wins.shape[0], _MIN_WIN_STRATEGY)
        optimal = np.full(history.wins.shape[0], int(SACRIFICE_TO_STRATEGY[3]))
        return np.where(ahead, sacrifice_all_but_one, optimal)


def default_adversaries() -> tuple[MemoryLHVAdversary, ...]:
    """The battery the adversarial self-test runs: one memoryless reference plus
    settings-memory, outcome-memory, and score-memory adversaries."""
    return (
        FixedSacrificeAdversary(),
        GreedyDenominatorAdversary(),
        WinStayLoseShiftAdversary(),
        QuitWhileAheadAdversary(),
    )


@dataclass(frozen=True)
class AdversaryRuns:
    """Results of ``trials`` independent ``rounds``-round games against one adversary.

    Carries per-setting tallies (not just pooled wins) so the naive per-setting
    correlator analysis can be evaluated on exactly the same data.
    """

    adversary: str
    rounds: int
    wins: np.ndarray  # (trials,) total wins
    setting_counts: np.ndarray  # (trials, 4)
    setting_wins: np.ndarray  # (trials, 4)


def _sealed(arr: np.ndarray) -> np.ndarray:
    """A read-only copy that owns its memory: no reference path (not even
    ``.base``) leads back to the referee's private ledger."""
    out = arr.copy()
    out.setflags(write=False)
    return out


def play_chsh_game(
    adversary: MemoryLHVAdversary,
    rounds: int,
    trials: int,
    *,
    seed: int = 0,
) -> AdversaryRuns:
    """Referee ``trials`` parallel CHSH games of ``rounds`` rounds each.

    Each round the adversary commits its strategies first (seeing only the
    history), then the referee draws settings uniformly at random from a
    *private* generator the adversary is never handed -- the adversary's own
    generator is spawned independently from the same seed, so peeking at,
    cloning, or exhausting it reveals nothing about the coming settings.
    Strategy indices are range-checked every round: any value in ``0..15`` is a
    genuine deterministic local strategy. The score lives in referee-private
    arrays; the adversary only ever sees frozen read-only snapshots of them,
    so it cannot write the ledger in either direction. Together the three
    checks make a passing run LHV with independently drawn settings by
    construction.
    """
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if trials <= 0:
        raise ValueError("trials must be positive")

    adversary_seq, referee_seq = np.random.SeedSequence(seed).spawn(2)
    adversary_rng = np.random.default_rng(adversary_seq)
    referee_rng = np.random.default_rng(referee_seq)

    # Referee-private ledger. Only _sealed() copies of these ever reach the
    # adversary; AdversaryRuns receives them after the last round is scored.
    setting_counts = np.zeros((trials, N_SETTINGS), dtype=np.int64)
    setting_wins = np.zeros((trials, N_SETTINGS), dtype=np.int64)
    wins = np.zeros(trials, dtype=np.int64)
    last_setting = np.full(trials, -1, dtype=np.int64)
    last_won = np.zeros(trials, dtype=bool)

    adversary.begin(trials, adversary_rng)
    rows = np.arange(trials)

    for round_index in range(rounds):
        history = GameHistory(
            rounds_played=round_index,
            setting_counts=_sealed(setting_counts),
            setting_wins=_sealed(setting_wins),
            wins=_sealed(wins),
            last_setting=_sealed(last_setting),
            last_won=_sealed(last_won),
        )
        strategies = np.asarray(adversary.strategies(history, adversary_rng))
        if strategies.shape != (trials,):
            raise ValueError(
                f"adversary returned shape {strategies.shape}, expected ({trials},)"
            )
        if not np.issubdtype(strategies.dtype, np.integer):
            raise ValueError("strategy indices must be integers")
        if strategies.min() < 0 or strategies.max() >= N_STRATEGIES:
            raise ValueError(
                "strategy index out of range 0..15: not a deterministic local strategy"
            )

        settings = referee_rng.integers(0, N_SETTINGS, size=trials)
        won = STRATEGY_WIN_TABLE[strategies, settings]

        setting_counts[rows, settings] += 1
        setting_wins[rows, settings] += won.astype(np.int64)
        wins += won.astype(np.int64)
        last_setting = settings
        last_won = won

    # Defence in depth: the ledger is private, so these invariants cannot fail
    # today -- but a refactor that leaked a live array would trip them loudly.
    if (
        int(wins.max()) > rounds
        or not np.array_equal(setting_wins.sum(axis=1), wins)
        or not np.array_equal(setting_counts.sum(axis=1), np.full(trials, rounds))
        or bool((setting_wins > setting_counts).any())
    ):  # pragma: no cover - unreachable while the ledger stays private
        raise RuntimeError("game ledger failed its integrity check")

    return AdversaryRuns(
        adversary=adversary.name,
        rounds=rounds,
        wins=wins,
        setting_counts=setting_counts,
        setting_wins=setting_wins,
    )
