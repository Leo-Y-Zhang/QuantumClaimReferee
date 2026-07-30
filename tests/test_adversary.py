import math

import numpy as np
import pytest

from minos.adversary import (
    SACRIFICE_TO_STRATEGY,
    STRATEGY_WIN_TABLE,
    FixedSacrificeAdversary,
    GreedyDenominatorAdversary,
    MemoryLHVAdversary,
    QuitWhileAheadAdversary,
    WinStayLoseShiftAdversary,
    default_adversaries,
    play_chsh_game,
)
from minos.chsh import CLASSICAL_WIN, chsh
from minos.power import certification_power, critical_wins
from minos.status import CERTIFIED, NOT_CERTIFIED, UNDERPOWERED

# ------------------------------------------------------------- strategy table


def test_win_table_shape_and_parity():
    # Deterministic local strategies win exactly 1 or 3 of the 4 setting pairs:
    # the four CHSH targets XOR to 1, so a deterministic strategy cannot satisfy
    # an even number of them.
    assert STRATEGY_WIN_TABLE.shape == (16, 4)
    per_strategy = STRATEGY_WIN_TABLE.sum(axis=1)
    assert set(per_strategy.tolist()) == {1, 3}


def test_classical_bound_emerges_from_the_table():
    # The 3/4 bound is derived, not assumed: the best deterministic strategy
    # wins exactly 3 of the 4 equally likely setting pairs.
    assert float(STRATEGY_WIN_TABLE.mean(axis=1).max()) == CLASSICAL_WIN


def test_sacrifice_map_wins_exactly_the_other_three():
    for s in range(4):
        row = STRATEGY_WIN_TABLE[SACRIFICE_TO_STRATEGY[s]]
        assert not row[s]
        assert int(row.sum()) == 3


# ------------------------------------------------------------------ simulator


def test_play_tallies_are_consistent():
    for adv in default_adversaries():
        runs = play_chsh_game(adv, 60, 50, seed=1)
        assert runs.rounds == 60
        np.testing.assert_array_equal(runs.setting_counts.sum(axis=1), np.full(50, 60))
        np.testing.assert_array_equal(runs.setting_wins.sum(axis=1), runs.wins)
        assert np.all(runs.setting_wins <= runs.setting_counts)
        assert np.all(runs.setting_wins >= 0)


def test_play_is_deterministic_given_seed():
    a = play_chsh_game(GreedyDenominatorAdversary(), 40, 30, seed=7)
    b = play_chsh_game(GreedyDenominatorAdversary(), 40, 30, seed=7)
    np.testing.assert_array_equal(a.wins, b.wins)
    np.testing.assert_array_equal(a.setting_wins, b.setting_wins)
    np.testing.assert_array_equal(a.setting_counts, b.setting_counts)


@pytest.mark.parametrize("kwargs", [{"rounds": 0, "trials": 5}, {"rounds": 5, "trials": 0}])
def test_play_rejects_nonpositive_sizes(kwargs):
    with pytest.raises(ValueError):
        play_chsh_game(FixedSacrificeAdversary(), **kwargs)


class _CheatingAdversary(MemoryLHVAdversary):
    """Returns an out-of-range strategy index: not a local strategy at all."""

    name = "cheater"

    def strategies(self, history, rng):
        return np.full(history.wins.shape[0], 99, dtype=np.int64)


def test_simulator_rejects_non_lhv_strategy_indices():
    # The LHV constraint is structural: every play must be one of the 16
    # deterministic local strategies, and the simulator enforces it.
    with pytest.raises(ValueError):
        play_chsh_game(_CheatingAdversary(), 5, 4, seed=0)


# ------------------------------------------------- adversary behaviour traits


def test_fixed_saturator_never_wins_its_sacrificed_setting():
    runs = play_chsh_game(FixedSacrificeAdversary(pair=3), 200, 40, seed=0)
    assert int(runs.setting_wins[:, 3].sum()) == 0
    np.testing.assert_array_equal(runs.setting_wins[:, :3], runs.setting_counts[:, :3])


def test_history_dependent_adversaries_spread_losses_across_settings():
    # Unlike the fixed saturator, the history-dependent adversaries move their
    # sacrificed setting around, so every setting eats losses somewhere.
    for adv in (GreedyDenominatorAdversary(), WinStayLoseShiftAdversary()):
        runs = play_chsh_game(adv, 400, 20, seed=0)
        losses = runs.setting_counts - runs.setting_wins
        assert np.all(losses.sum(axis=0) > 0), adv.name


def test_quit_while_ahead_wins_strictly_less_than_the_bound():
    # Whenever it is ahead of the 3/4 pace it throws rounds (conditional win
    # probability 1/4), so its overall win rate sits strictly below 3/4.
    n, trials = 400, 400
    runs = play_chsh_game(QuitWhileAheadAdversary(lead=2), n, trials, seed=0)
    mean_rate = float(runs.wins.mean()) / n
    mc_sigma = math.sqrt(CLASSICAL_WIN * (1.0 - CLASSICAL_WIN) / (n * trials))
    assert mean_rate < CLASSICAL_WIN - 5.0 * mc_sigma


# --------------------------------------------- the validity theorem, verified


def test_no_adversary_exceeds_the_exact_binomial_ceiling():
    # THE point of the adversarial self-test: for ANY memory-LHV adversary the
    # certification (false-positive) rate is capped by the exact binomial tail
    # P[Bin(n, 3/4) >= c_alpha(n)] -- computed by the same power machinery that
    # minos plan uses -- because total wins are stochastically dominated by
    # Binomial(n, 3/4) even when outcomes depend on past settings and outcomes.
    n, trials, alpha = 80, 4000, 0.05
    c = critical_wins(n, alpha)
    ceiling = certification_power(n, CLASSICAL_WIN, alpha=alpha)
    assert ceiling <= alpha  # the ceiling itself is calibrated
    tol = 4.0 * math.sqrt(ceiling * (1.0 - ceiling) / trials)
    for adv in default_adversaries():
        runs = play_chsh_game(adv, n, trials, seed=3)
        emp = float(np.mean(runs.wins >= c))
        assert emp <= ceiling + tol, adv.name


def test_sacrifice_adversaries_match_the_binomial_exactly():
    # Any history-measurable choice of sacrificed setting keeps the conditional
    # win probability at exactly 3/4, so total wins are Binomial(n, 3/4) in
    # distribution: the mean and the certification tail must MATCH the exact
    # values, not merely stay below them. Memory moves per-setting statistics,
    # never the pooled win count -- which is exactly why minos certifies from
    # pooled wins.
    n, trials, alpha = 80, 4000, 0.05
    c = critical_wins(n, alpha)
    ceiling = certification_power(n, CLASSICAL_WIN, alpha=alpha)
    mean_tol = 4.0 * math.sqrt(CLASSICAL_WIN * (1.0 - CLASSICAL_WIN) / (n * trials))
    tail_tol = 4.0 * math.sqrt(ceiling * (1.0 - ceiling) / trials)
    for adv in (
        FixedSacrificeAdversary(),
        GreedyDenominatorAdversary(),
        WinStayLoseShiftAdversary(),
    ):
        runs = play_chsh_game(adv, n, trials, seed=5)
        mean_rate = float(runs.wins.mean()) / n
        assert abs(mean_rate - CLASSICAL_WIN) < mean_tol, adv.name
        emp = float(np.mean(runs.wins >= c))
        assert abs(emp - ceiling) <= tail_tol, adv.name


# ------------------------------------------- ties into the verdict machinery


def test_verdicts_on_adversarial_runs_match_the_shipped_chsh_machinery():
    # For every distinct win count an adversary actually produced, the shipped
    # chsh() verdict must equal the classification the self-test uses:
    # CERTIFIED iff wins >= c_alpha(n); else NOT_CERTIFIED iff omega <= 3/4;
    # else UNDERPOWERED.
    n = 40
    c = critical_wins(n, 0.05)
    runs = play_chsh_game(GreedyDenominatorAdversary(), n, 250, seed=4)
    for w in np.unique(runs.wins):
        w = int(w)
        expected = CERTIFIED if w >= c else (NOT_CERTIFIED if 4 * w <= 3 * n else UNDERPOWERED)
        r = chsh(w, n, alpha=0.05, setting_randomness_declared=True)
        assert r.status == expected, f"wins={w}"


def test_underpowered_adversarial_run_carries_the_plan_hint():
    # An adversary at the bound lands UNDERPOWERED about half the time; those
    # verdicts must carry the minos plan hint (exact-binomial rounds-for-power),
    # exactly as on real data.
    n = 80
    c = critical_wins(n, 0.05)
    runs = play_chsh_game(FixedSacrificeAdversary(), n, 200, seed=6)
    underpowered = [int(w) for w in runs.wins if 4 * int(w) > 3 * n and int(w) < c]
    assert underpowered  # the adversary does produce such runs
    r = chsh(underpowered[0], n, alpha=0.05, setting_randomness_declared=True)
    assert r.status == UNDERPOWERED
    assert "rounds for power" in r.summary()
