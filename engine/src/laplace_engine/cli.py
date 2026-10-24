"""Command line: freeze a release, seal an issue, verify an issue."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from laplace_engine import candles, issue, release, verify
from laplace_engine.issue import Ledger

ROBINHOOD_CHAIN = 4663
LEDGER = "0x15AaE63A0EF391559153bCF9c56E5E2BE916b2BE"

log = logging.getLogger("laplace_engine")


def issue_dir(data: Path, number: int) -> Path:
    return data / "issues" / str(number)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laplace-engine", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("release", help="freeze a research run into a release")
    freeze.add_argument("--run", type=Path, required=True, help="research run directory (with report.json)")
    freeze.add_argument("--name", required=True)
    freeze.add_argument("--out", type=Path, required=True, help="release directory (public part, in the repo)")
    freeze.add_argument("--private", type=Path, required=True, help="where private program sources are kept")

    seal = sub.add_parser("issue", help="seal the current (or a given) issue into local files")
    seal.add_argument("--release", type=Path, required=True)
    seal.add_argument("--private", type=Path, required=True)
    seal.add_argument("--data", type=Path, required=True, help="engine data directory (candle cache, issues)")
    seal.add_argument("--issue", type=int, help="issue number (hours since the epoch); default: the current hour")
    seal.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    seal.add_argument("--wait-minutes", type=float, default=10.0, help="how long to wait for the candles")
    seal.add_argument("--ledger", default=LEDGER)

    check = sub.add_parser("verify", help="rebuild an issue's leaves, root and pools from its files")
    check.add_argument("--data", type=Path, required=True)
    check.add_argument("--issue", type=int, required=True)
    return parser


def cmd_release(args: argparse.Namespace) -> int:
    from laplace_engine.freeze import freeze

    frozen = freeze(args.run, args.name)
    release.write(frozen, args.out, args.private)
    for entry in frozen.describe():
        print(f"{entry['name']:12s} {entry['kind']:8s} {entry['id']}")
    return 0


def cmd_issue(args: argparse.Namespace) -> int:
    started = time.monotonic()
    number = args.issue if args.issue is not None else candles.current_issue()
    target = issue_dir(args.data, number)
    if (target / "manifest.json").exists():
        log.info("issue %d is already sealed in %s", number, target)
        return 0
    rel = release.load(args.release, args.private)
    args.data.mkdir(parents=True, exist_ok=True)
    # A live issue waits for its candles until the deadline; a past issue (a dry run) only needs a moment.
    deadline = max(
        candles.issue_time(number) + timedelta(minutes=args.wait_minutes), datetime.now(UTC) + timedelta(minutes=1)
    )
    panel = candles.wait_for(number, args.data / "candles", deadline=deadline)
    with tempfile.TemporaryDirectory(dir=args.data) as workdir:
        finals, failures = issue.run_release(rel, panel, Path(workdir), workers=args.workers)
    sealed = issue.seal(
        number, rel, panel, finals, failures, ledger=Ledger(ROBINHOOD_CHAIN, args.ledger), commit=issue.engine_commit()
    )
    issue.write_atomic(target / "forecasts.json", sealed.forecasts)
    issue.write_atomic(target / "manifest.json", sealed.manifest)
    for name, error in sealed.failures.items():
        log.warning("issue %d: %s failed: %s", number, name, error)
    root, manifest = sealed.root.hex(), sealed.manifest_hash.hex()
    seconds = time.monotonic() - started
    print(f"issue {number}: {sealed.leaves} forecasts, root 0x{root}, manifest 0x{manifest} ({seconds:.1f}s)")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    target = issue_dir(args.data, args.issue)
    result = verify.verify((target / "manifest.json").read_bytes(), (target / "forecasts.json").read_bytes())
    print(
        f"issue {result.issue}: {result.leaves} forecasts and {result.pools_checked} pool(s) check out; "
        f"root 0x{result.root.hex()}, manifest 0x{result.manifest_hash.hex()}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    args = build_parser().parse_args(argv)
    commands = {"release": cmd_release, "issue": cmd_issue, "verify": cmd_verify}
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
