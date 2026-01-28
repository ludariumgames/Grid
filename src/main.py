from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from reports import build_report, print_report, save_report_json
from simulator import SimulationInput, run_simulations


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="grid-sim",
        description="Console simulator for the Grid drafting game (config-driven).",
    )

    p.add_argument(
        "--config",
        required=True,
        help="Path to JSON config, e.g. config/game_rules_config_v0_1.json",
    )
    p.add_argument(
        "--players",
        type=int,
        required=True,
        help="Number of players (2..4)",
    )
    p.add_argument(
        "--games",
        type=int,
        required=True,
        help="Number of games to simulate (e.g. 10000..100000)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility",
    )
    p.add_argument(
        "--agents",
        nargs="+",
        required=True,
        help="Agent names. Provide 1 name to use for all players, or exactly N names. Example: --agents random",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional path to save JSON report, e.g. results/results.json",
    )

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    inp = SimulationInput(
        config_path=args.config,
        num_players=args.players,
        games=args.games,
        seed=args.seed,
        agent_names=list(args.agents),
    )

    try:
        _cfg, agg = run_simulations(inp)
        report = build_report(inp, agg)
        print_report(report)

        if args.out:
            save_report_json(args.out, report)
            print("\nsaved to", args.out)

        return 0

    except Exception as e:
        print("ERROR:", str(e))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
