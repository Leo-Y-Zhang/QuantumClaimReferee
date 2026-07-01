"""``qref`` command-line interface.

Exit code 0 means CERTIFIED, non-zero means anything else -- so qref drops into a
CI gate or a research pipeline exactly like a test runner.
"""

from __future__ import annotations

import argparse
import sys

from ._version import __version__
from .chsh import chsh
from .selftest import binomial_interval_coverage, chsh_null_false_positive_rates
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


def _cmd_selftest(args: argparse.Namespace) -> int:
    n = args.n
    if n <= 0:
        raise ValueError("--n must be a positive integer")
    if args.trials <= 0:
        raise ValueError("--trials must be a positive integer")
    print(f"CHSH null false-positive rates at n={n} (nominal alpha=0.05):")
    fpr = chsh_null_false_positive_rates(n, trials=args.trials)
    for name, rate in fpr.items():
        flag = ""
        if name == "memory_robust":
            flag = "  <- valid" if rate <= 0.053 else "  <- CHECK"
        if name == "naive_observed":
            flag = "  <- INVALID (inflated)" if rate > 0.055 else ""
        print(f"  {name:<16}: {rate:.4f}{flag}")
    print(f"\nWilson interval coverage at p=0.9, n={n} (nominal 0.95):")
    cov = binomial_interval_coverage("wilson", 0.9, n, trials=args.trials)
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
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qref", description=__doc__)
    parser.add_argument("--version", action="version", version=f"qreferee {__version__}")
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

    p_self = sub.add_parser("selftest", help="run coverage / calibration self-tests")
    p_self.add_argument("--n", type=int, default=80)
    p_self.add_argument("--trials", type=int, default=100_000)
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
