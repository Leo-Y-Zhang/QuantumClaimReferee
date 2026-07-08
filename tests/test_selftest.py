from minos.selftest import binomial_interval_coverage, chsh_null_false_positive_rates


def test_naive_chsh_is_miscalibrated_at_small_n():
    # The thesis of the whole package, checked empirically.
    fpr = chsh_null_false_positive_rates(80, trials=60_000, seed=0)
    assert fpr["memory_robust"] <= 0.055          # rigorous test is valid
    assert fpr["naive_observed"] > 0.055          # naive test over-certifies
    assert fpr["naive_observed"] > fpr["memory_robust"]


def test_naive_and_rigorous_agree_at_large_n():
    fpr = chsh_null_false_positive_rates(8000, trials=60_000, seed=0)
    assert abs(fpr["naive_observed"] - fpr["memory_robust"]) < 0.02


def test_clopper_pearson_is_conservative():
    cov = binomial_interval_coverage("clopper-pearson", 0.5, 100, trials=20_000, seed=1)
    assert cov >= 0.95


def test_wilson_coverage_near_nominal():
    cov = binomial_interval_coverage("wilson", 0.3, 200, trials=20_000, seed=2)
    assert cov >= 0.93
