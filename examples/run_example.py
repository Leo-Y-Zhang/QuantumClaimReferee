"""Worked example: the naive-vs-rigorous CHSH wedge, and a referee report.

Fully offline, but not import-free: qcref lives under src/ and pulls in numpy and
scipy, so install first.  python -m pip install -e .  then  python examples/run_example.py
"""

from __future__ import annotations

from qcref import chsh, plan_rounds, referee_report, s_to_omega
from qcref.selftest import chsh_null_false_positive_rates
from qcref.status import ASSUMPTIONS_UNMET
from qcref.verdict import Study


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
    # Both the p-value and the label are read off the result, and the CHSH status
    # is carried into the study rather than stepped around: a hypothesis whose data
    # are unphysical has to reach the report as ASSUMPTIONS_UNMET, not as a tiny
    # p-value. A hand-typed estimate can drift past the Tsirelson bound while the
    # p-value it sits next to stays small; a derived one cannot.
    study = Study(alpha=0.05, correction="holm")
    study.add("ghz3_fidelity_gt_0.90", 6.0e-4, estimate="F=0.947 [0.921, 0.966]")
    pair_2_3 = chsh(1700, 2000, setting_randomness_declared=True)
    study.add(
        "pair_2_3_entangled_chsh",
        pair_2_3.p_memory_robust,
        assumptions_met=pair_2_3.status != ASSUMPTIONS_UNMET,
        estimate=f"S={pair_2_3.S:.2f}",
    )
    pair_1_2 = chsh(84, 100, setting_randomness_declared=True)
    study.add(
        "pair_1_2_entangled_chsh",
        pair_1_2.p_memory_robust,
        assumptions_met=pair_1_2.status != ASSUMPTIONS_UNMET,
        estimate=f"S={pair_1_2.S:.2f} (few shots)",
    )
    print(referee_report(study.run(), title="ghz3_device_run"))

    print("\n=== 5) Plan the next run instead of guessing ===")
    print("Rounds needed to certify a hypothesised S=2.4 at alpha=0.05, 90% power:")
    print(plan_rounds(s_to_omega(2.4), alpha=0.05, power=0.9).summary())


if __name__ == "__main__":
    main()
