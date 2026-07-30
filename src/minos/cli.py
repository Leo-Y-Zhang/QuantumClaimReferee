"""``minos`` command-line interface.

Exit code 0 means CERTIFIED, non-zero means anything else -- so minos drops into a
CI gate or a research pipeline exactly like a test runner.
"""

from __future__ import annotations

import argparse
import sys

from ._version import __version__
from .chsh import CLASSICAL_WIN, chsh, s_to_omega
from .power import DEFAULT_MAX_ROUNDS, certification_power, critical_wins, plan_rounds
from .selftest import (
    binomial_interval_coverage,
    chsh_adversarial_false_positive_rates,
    chsh_null_false_positive_rates,
)
from .verdict import Study


def _cmd_chsh(args: argparse.Namespace) -> int:
    result = chsh(
        args.wins,
        args.rounds,
        alpha=args.alpha,
        setting_randomness_declared=args.randomised,
    )
    print(result.summary())
    return 0 if result.certified else 1


def _cmd_plan(args: argparse.Namespace) -> int:
    win_rate = args.win_rate if args.win_rate is not None else s_to_omega(args.S)
    plan = plan_rounds(
        win_rate,
        alpha=args.alpha,
        power=args.power,
        max_rounds=args.max_rounds,
    )
    print(plan.summary())
    return 0


def _cmd_selftest(args: argparse.Namespace) -> int:
    n = args.n
    if n <= 0:
        raise ValueError("--n must be a positive integer")
    # the sequential adversarial games cost more per trial than the vectorised
    # null sweep, so the two modes get different default trial counts
    trials = args.trials if args.trials is not None else (2_000 if args.adversarial else 100_000)
    if trials <= 0:
        raise ValueError("--trials must be a positive integer")
    if args.adversarial:
        return _selftest_adversarial(n, trials)
    return _selftest_null(n, trials)


def _selftest_adversarial(n: int, trials: int) -> int:
    alpha = 0.05
    c = critical_wins(n, alpha)
    ceiling = certification_power(n, CLASSICAL_WIN, alpha=alpha)
    print(
        f"Adversarial memory-loophole self-test: n={n} rounds, {trials} runs per "
        f"adversary (alpha={alpha})"
    )
    print(
        f"  exact ceiling for ANY memory-LHV adversary: "
        f"P[Bin({n}, 3/4) >= {c}] = {ceiling:.4f}"
    )
    print(
        "  (outcomes may depend on all past settings and outcomes; the game tail"
        " must stay at or below the ceiling, the naive per-setting sigma test"
        " has no such guarantee)\n"
    )
    for report in chsh_adversarial_false_positive_rates(n, alpha=alpha, trials=trials).values():
        print(report.summary())
        print()
    print(
        "note: more rounds cannot rescue a true LHV source -- minos plan refuses\n"
        "      win rates at or below 3/4 (the certify probability never exceeds\n"
        "      alpha there), and UNDERPOWERED verdicts on such data price a\n"
        "      follow-up that will again fail with probability >= 1 - alpha."
    )
    return 0


def _selftest_null(n: int, trials: int) -> int:
    print(f"CHSH null false-positive rates at n={n} (nominal alpha=0.05):")
    fpr = chsh_null_false_positive_rates(n, trials=trials)
    for name, rate in fpr.items():
        flag = ""
        if name == "memory_robust":
            flag = "  <- valid" if rate <= 0.053 else "  <- CHECK"
        if name == "naive_observed":
            flag = "  <- INVALID (inflated)" if rate > 0.055 else ""
        print(f"  {name:<16}: {rate:.4f}{flag}")
    print(f"\nWilson interval coverage at p=0.9, n={n} (nominal 0.95):")
    cov = binomial_interval_coverage("wilson", 0.9, n, trials=trials)
    print(f"  wilson          : {cov:.4f}")
    return 0


def _cmd_demo(_args: argparse.Namespace) -> int:
    print("Scarce data (n=80, 66 wins), settings randomised:")
    r = chsh(66, 80, setting_randomness_declared=True)
    print(r.summary())
    print("\nSame data, plenty of it (n=8000, 6400 wins):")
    print(chsh(6400, 8000, setting_randomness_declared=True).summary())
    print("\nBest-of-6 scan, best rigorous p=0.03:")
    study = Study(alpha=0.05, correction="holm")
    for i, p in enumerate([0.03, 0.21, 0.44, 0.61, 0.77, 0.90]):
        study.add(f"pair_{i}", p)
    print(study.run().summary())
    print("\nPlanning ahead: rounds needed to certify S=2.4 with 90% power:")
    print(plan_rounds(s_to_omega(2.4), alpha=0.05, power=0.9).summary())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minos", description=__doc__)
    parser.add_argument("--version", action="version", version=f"minos {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_chsh = sub.add_parser("chsh", help="certify a CHSH violation from wins/rounds")
    p_chsh.add_argument("--wins", type=int, required=True)
    p_chsh.add_argument("--rounds", type=int, required=True)
    p_chsh.add_argument("--alpha", type=float, default=0.05)
    p_chsh.add_argument(
        "--randomised",
        action="store_true",
        help="affirm settings were randomised per round (required to certify)",
    )
    p_chsh.set_defaults(func=_cmd_chsh)

    p_plan = sub.add_parser(
        "plan",
        help="exact power analysis: minimal rounds to certify at alpha with target power",
    )
    hypo = p_plan.add_mutually_exclusive_group(required=True)
    hypo.add_argument("--S", type=float, help="hypothesised CHSH value (2 < S <= 2*sqrt(2))")
    hypo.add_argument(
        "--win-rate",
        type=float,
        dest="win_rate",
        help="hypothesised per-round win probability (3/4 < p <= 1)",
    )
    p_plan.add_argument("--alpha", type=float, default=0.05)
    p_plan.add_argument("--power", type=float, default=0.9)
    p_plan.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    p_plan.set_defaults(func=_cmd_plan)

    p_self = sub.add_parser("selftest", help="run coverage / calibration self-tests")
    p_self.add_argument("--n", type=int, default=80)
    p_self.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Monte-Carlo runs (default 100000, or 2000 with --adversarial)",
    )
    p_self.add_argument(
        "--adversarial",
        action="store_true",
        help=(
            "referee history-dependent memory-LHV adversaries (outcomes may depend"
            " on past settings and outcomes) and check every one stays bounded"
        ),
    )
    p_self.set_defaults(func=_cmd_selftest)

    p_demo = sub.add_parser("demo", help="print the worked wedge example")
    p_demo.set_defaults(func=_cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as exc:
        # Surface bad input as a clean usage error (exit 2), not a traceback.
        parser.error(str(exc))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
