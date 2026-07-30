"""Worked example: the naive-vs-rigorous CHSH wedge, and a referee report.

Fully offline. Run:  python examples/run_example.py
"""

from __future__ import annotations

from minos import chsh, plan_rounds, referee_report, s_to_omega
from minos.selftest import chsh_null_false_positive_rates
from minos.verdict import Study


def main() -> None:
    print("=== 1) Scarce data: naive certifies, rigorous refuses ===")
    r = chsh(66, 80, setting_randomness_declared=True)
    print(r.summary())

    print("\n=== 2) Plenty of data: everyone agrees ===")
    print(chsh(6400, 8000, setting_randomness_declared=True).summary())

    print("\n=== 3) Why: the naive test is miscalibrated under the null ===")
    for n in (80, 8000):
        fpr = chsh_null_false_positive_rates(n, trials=100_000)
        print(
            f"  n={n:<5} false-positive rate  "
            f"naive(observed)={fpr['naive_observed']:.4f}  "
            f"rigorous(game tail)={fpr['memory_robust']:.4f}   (nominal 0.05)"
        )

    print("\n=== 4) A referee report over several hypotheses ===")
    study = Study(alpha=0.05, correction="holm")
    study.add("ghz3_fidelity_gt_0.90", 6.0e-4, estimate="F=0.947 [0.921, 0.966]")
    study.add(
        "pair_2_3_entangled_chsh",
        chsh(1720, 2000, setting_randomness_declared=True).p_memory_robust,
        estimate="S=2.88",
    )
    study.add(
        "pair_1_2_entangled_chsh",
        chsh(84, 100, setting_randomness_declared=True).p_memory_robust,
        estimate="S=2.72 (few shots)",
    )
    print(referee_report(study.run(), title="ghz3_device_run"))

    print("\n=== 5) Plan the next run instead of guessing ===")
    print("Rounds needed to certify a hypothesised S=2.4 at alpha=0.05, 90% power:")
    print(plan_rounds(s_to_omega(2.4), alpha=0.05, power=0.9).summary())


if __name__ == "__main__":
    main()
