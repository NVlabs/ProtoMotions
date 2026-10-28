# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line entry point: ``python -m protomotions.tests.parity ...``.

Exit codes of ``compare``: 0 identical, 1 value divergence, 2 call order
changed (values identical), 3 the comparison itself failed.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import replace
from typing import Optional, Sequence

# ``compare`` and ``list`` stay light: the scenario catalogue is imported on
# demand and the recorder (with the simulator and evaluator stacks) only when
# recording.
from protomotions.tests.parity.compare import (
    compare,
    format_result,
    load_trace,
    save_trace,
)
from protomotions.tests.parity.mutations import MUTATIONS

EXIT_IDENTICAL = 0
EXIT_VALUE_DIVERGENCE = 1
EXIT_CALL_ORDER_CHANGED = 2
EXIT_COMPARE_FAILED = 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m protomotions.tests.parity",
        description="Record and compare BaseEnv parity traces.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    rec = commands.add_parser("record", help="record one scenario to a .pt file")
    rec.add_argument("--scenario", required=True, help="a name from `list --names`")
    rec.add_argument("--out", required=True)
    rec.add_argument("--steps", type=int, default=None)
    rec.add_argument("--mutation", default=None, choices=sorted(MUTATIONS))
    rec.add_argument("--simulator", default=None, help="GPU mode: backend name")
    rec.add_argument("--num-envs", type=int, default=None, help="GPU mode: env count")

    cmp = commands.add_parser(
        "compare",
        help="compare two traces; exit 1 on a value divergence, 2 on call order",
    )
    cmp.add_argument("a")
    cmp.add_argument("b")
    cmp.add_argument("--atol", type=float, default=0.0)
    cmp.add_argument("--ignore", nargs="*", default=[], metavar="PREFIX")

    lst = commands.add_parser("list", help="list the scenarios")
    lst.add_argument("--names", action="store_true", help="names only, one per line")
    return parser


def _record(args) -> int:
    from protomotions.tests.parity.recorder import record, summarize
    from protomotions.tests.parity.scenarios import SCENARIOS

    if args.scenario not in SCENARIOS:
        print(
            f"unknown scenario {args.scenario!r}; choose from "
            f"{', '.join(sorted(SCENARIOS))}",
            file=sys.stderr,
        )
        return 2
    scenario = SCENARIOS[args.scenario]
    overrides = {}
    if args.steps is not None:
        overrides["steps"] = args.steps
    if args.simulator is not None:
        overrides["simulator"] = args.simulator
    if args.num_envs is not None:
        overrides["num_envs"] = args.num_envs
    if overrides:
        scenario = replace(scenario, **overrides)
    trace = record(scenario, mutation=args.mutation)
    save_trace(trace, args.out)
    print({**summarize(trace), "out": args.out})
    return 0


def _compare(args) -> int:
    try:
        result = compare(
            load_trace(args.a),
            load_trace(args.b),
            atol=args.atol,
            ignore_prefixes=tuple(args.ignore),
        )
    except Exception:  # the verdict must never be mistaken for a divergence
        traceback.print_exc()
        print(f"COMPARE FAILED: {sys.exc_info()[1]!r}")
        return EXIT_COMPARE_FAILED
    print(format_result(result, atol=args.atol))
    if result.identical:
        return EXIT_IDENTICAL
    if result.values_identical:
        return EXIT_CALL_ORDER_CHANGED
    return EXIT_VALUE_DIVERGENCE


def _list(args) -> int:
    from protomotions.tests.parity.scenarios import SCENARIOS

    for name, scenario in SCENARIOS.items():
        if args.names:
            print(name)
            continue
        experiment = scenario.experiment or "fake simulator (tests/support)"
        flags = [
            flag
            for flag, enabled in (
                ("stress", scenario.stress),
                ("g1_no_mesh", scenario.g1_no_mesh),
                (
                    "body_mass_randomization=False",
                    not scenario.body_mass_randomization,
                ),
                (f"contact_bodies={scenario.contact_bodies}", scenario.contact_bodies),
                (f"fake_motions={scenario.fake_motions}", scenario.fake_motions),
                (
                    f"fake_target_checks={','.join(scenario.fake_target_checks)}",
                    scenario.fake_target_checks,
                ),
            )
            if enabled
        ]
        print(
            f"{name:16s} {experiment}  robot={scenario.robot} "
            f"num_envs={scenario.num_envs} steps={scenario.steps}"
            + (f" [{', '.join(flags)}]" if flags else "")
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    return {"record": _record, "compare": _compare, "list": _list}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
