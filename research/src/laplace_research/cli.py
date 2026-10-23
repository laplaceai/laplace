"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from laplace_research import __version__

RUNS = Path(__file__).resolve().parents[2] / "runs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laplace-research")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    majors = sub.add_parser("majors", help="blinded Majors backtest (PoC 1)")
    majors.add_argument("action", choices=["run", "report"])
    majors.add_argument("--run-id", required=True)
    majors.add_argument("--days", type=int, default=30)
    majors.add_argument("--step-hours", type=int, default=6)
    majors.add_argument("--concurrency", type=int, default=16)
    majors.add_argument("--agents", help="comma-separated model/directive ids (default: all)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "majors":
        from laplace_research.experiments import majors, report

        run_dir = RUNS / args.run_id
        if args.action == "run":
            agents = majors.default_agents()
            if args.agents:
                wanted = set(args.agents.split(","))
                agents = [a for a in agents if a.agent_id in wanted]
            majors.run(
                run_dir,
                days=args.days,
                step_hours=args.step_hours,
                concurrency=args.concurrency,
                agents=agents,
            )
        print(report.render(report.report(run_dir)))
    return 0
