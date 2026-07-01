import pytest

from qreferee.chsh import chsh, wins_from_setting_counts


def test_value_map_local_bound():
    r = chsh(75, 100, setting_randomness_declared=True)  # omega = 0.75
    assert r.S == pytest.approx(2.0)


def test_regime_a_scarce_data_underpowered():
    r = chsh(66, 80, setting_randomness_declared=True)
    assert r.S == pytest.approx(2.6, abs=1e-9)
    assert r.p_memory_robust == pytest.approx(0.074, abs=5e-3)
    assert r.p_naive_observed == pytest.approx(0.0387, abs=5e-3)
    # naive certifies (<0.05) where the rigorous bound does not -> the whole point
    assert r.p_naive_observed < 0.05 < r.p_memory_robust
    assert r.status == "UNDERPOWERED"


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
        (1, 1): {(0, 1): 7, (1, 0): 2, (0, 0): 1},  # win iff a!=b -> 9 wins / 10
    }
    wins, rounds = wins_from_setting_counts(counts)
    assert (wins, rounds) == (15 + 9, 18 + 10)


@pytest.mark.parametrize("wins,rounds", [(-1, 10), (11, 10), (5, 0)])
def test_invalid_inputs_raise(wins, rounds):
    with pytest.raises(ValueError):
        chsh(wins, rounds)
