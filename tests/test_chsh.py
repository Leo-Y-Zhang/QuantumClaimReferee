import math

import pytest

from minos.chsh import (
    TSIRELSON_S,
    chsh,
    game_tail_pvalue,
    omega_to_s,
    s_to_omega,
    wins_from_setting_counts,
)


def test_value_map_local_bound():
    r = chsh(75, 100, setting_randomness_declared=True)  # omega = 0.75
    assert r.S == pytest.approx(2.0)


def test_omega_s_mapping_round_trips():
    assert omega_to_s(0.75) == pytest.approx(2.0)
    assert s_to_omega(2.0) == pytest.approx(0.75)
    assert s_to_omega(TSIRELSON_S) == pytest.approx((2.0 + math.sqrt(2.0)) / 4.0)
    for s in (2.0, 2.4, 2.7, TSIRELSON_S):
        assert omega_to_s(s_to_omega(s)) == pytest.approx(s)


def test_game_tail_pvalue_is_the_shipped_pvalue():
    # The exported tail must be byte-identical to what chsh() reports and decides on.
    for wins, rounds in [(66, 80), (6400, 8000), (0, 5), (5, 5)]:
        r = chsh(wins, rounds, setting_randomness_declared=True)
        assert game_tail_pvalue(wins, rounds) == r.p_memory_robust


def test_game_tail_pvalue_hand_computed():
    # P[Bin(2, 3/4) >= 1] = 1 - (1/4)^2 = 15/16; P[Bin(3, 3/4) >= 3] = (3/4)^3.
    assert game_tail_pvalue(1, 2) == pytest.approx(15.0 / 16.0)
    assert game_tail_pvalue(3, 3) == pytest.approx(0.75**3)
    assert game_tail_pvalue(0, 4) == pytest.approx(1.0)


def test_regime_a_scarce_data_underpowered():
    r = chsh(66, 80, setting_randomness_declared=True)
    assert r.S == pytest.approx(2.6, abs=1e-9)
    assert r.p_memory_robust == pytest.approx(0.074, abs=5e-3)
    assert r.p_naive_observed == pytest.approx(0.0387, abs=5e-3)
    # naive certifies (<0.05) where the rigorous bound does not -> the whole point
    assert r.p_naive_observed < 0.05 < r.p_memory_robust
    assert r.status == "UNDERPOWERED"


def test_underpowered_summary_states_rounds_needed():
    # plan_rounds(0.825, alpha=0.05, power=0.9) -> 255 rounds; the UNDERPOWERED
    # summary must surface that hint, honestly labelled with the assumed power.
    r = chsh(66, 80, setting_randomness_declared=True)
    text = r.summary()
    assert "255" in text
    assert "90% power" in text
    assert "observed" in text


def test_certified_summary_has_no_plan_hint():
    text = chsh(6400, 8000, setting_randomness_declared=True).summary()
    assert "90% power" not in text


def test_azuma_is_looser_than_exact():
    r = chsh(66, 80, setting_randomness_declared=True)
    assert r.p_azuma > r.p_memory_robust


def test_regime_b_plenty_of_data_certified():
    r = chsh(6400, 8000, setting_randomness_declared=True)
    assert r.status == "CERTIFIED"
    assert r.certified
    assert r.p_memory_robust < 1e-10


def test_decisive_non_violation_is_not_certified_not_underpowered():
    # ample data, clearly below the local bound: NOT_CERTIFIED, not "get more shots"
    r = chsh(100, 1000, setting_randomness_declared=True)  # omega=0.1, S<2
    assert r.status == "NOT_CERTIFIED"


def test_default_deny_without_randomness_declaration():
    r = chsh(6400, 8000)  # randomness not declared
    assert r.status == "ASSUMPTIONS_UNMET"
    assert not r.certified


@pytest.mark.parametrize("kwargs", [{"alpha": 1.5}, {"alpha": 0.0}, {"level": 1.0}])
def test_out_of_range_alpha_or_level_raise(kwargs):
    with pytest.raises(ValueError):
        chsh(6400, 8000, setting_randomness_declared=True, **kwargs)


def test_wins_from_setting_counts_rejects_out_of_domain_keys():
    with pytest.raises(ValueError):
        wins_from_setting_counts({(0, 2): {(0, 0): 5}})
    with pytest.raises(ValueError):
        wins_from_setting_counts({(0, 0): {(0, 3): 5}})


def test_wins_from_setting_counts():
    counts = {
        (0, 0): {(0, 0): 10, (1, 1): 5, (0, 1): 3},  # win iff a==b -> 15 wins / 18
        (0, 1): {(0, 0): 4, (1, 0): 2},  # win iff a==b -> 4 wins / 6
        (1, 0): {(1, 1): 6, (0, 1): 1},  # win iff a==b -> 6 wins / 7
        (1, 1): {(0, 1): 7, (1, 0): 2, (0, 0): 1},  # win iff a!=b -> 9 wins / 10
    }
    wins, rounds = wins_from_setting_counts(counts)
    assert (wins, rounds) == (15 + 4 + 6 + 9, 18 + 6 + 7 + 10)


def test_an_incomplete_settings_dict_is_refused():
    """This used to be asserted as WORKING, which is how the hole survived.

    The CHSH game is defined over all four input pairs, so a run that never
    presented one has not played it. Pooling it anyway lets a dropped job turn a
    purely classical device into a certified violation: with only (0,0) present,
    answering a=b=0 every round wins every round.
    """
    for partial in (
        {(0, 0): {(0, 0): 10}},
        {(0, 0): {(0, 0): 10}, (1, 1): {(0, 1): 7}},
        {(0, 0): {(0, 0): 1}, (0, 1): {(0, 0): 1}, (1, 0): {(0, 0): 1}},
    ):
        with pytest.raises(ValueError, match="all four input pairs"):
            wins_from_setting_counts(partial)


def test_certification_at_small_n_would_require_impossible_physics():
    """Documents an OPEN issue rather than asserting a fix, deliberately.

    TSIRELSON_S is defined and exported by this module and guards nothing, so a
    run reporting S above it is still certified. Refusing that outright is the
    obviously right thing, but it collides with minos.power: for small n the
    win count needed to certify implies an S no quantum device can reach, so
    critical_wins currently advises a threshold that is impossible to meet
    honestly. Fixing the guard without fixing the planner would leave the tool
    recommending unreachable run sizes.

    This test pins the arithmetic so the decision is made on numbers rather than
    from memory. It asserts what IS true today, including the hole.
    """
    from minos.power import critical_wins

    impossible_at = [
        n
        for n in (11, 40, 100, 400)
        if omega_to_s(critical_wins(n, alpha=0.05) / n) > TSIRELSON_S
    ]
    assert impossible_at == [11, 40], (
        "certifying at these round counts requires S above the Tsirelson bound; "
        "if this list changes, minos.power gained (or lost) a physical floor"
    )

    # and the hole itself: an always-win device is certified today
    always_win = chsh(wins=2000, rounds=2000, setting_randomness_declared=True)
    assert always_win.S == 4.0 > TSIRELSON_S
    assert always_win.certified, (
        "if this now fails, the Tsirelson guard has been added - good; update "
        "this test and check minos.power refuses n below the physical floor"
    )


@pytest.mark.parametrize("wins,rounds", [(-1, 10), (11, 10), (5, 0)])
def test_invalid_inputs_raise(wins, rounds):
    with pytest.raises(ValueError):
        chsh(wins, rounds)
