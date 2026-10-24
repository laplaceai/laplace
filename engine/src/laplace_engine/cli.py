"""Command line: freeze a release; seal, commit and publish an issue; verify an issue; show recent issues."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from eth_hash.auto import keccak

from laplace_engine import candles, chain, issue, publish, release, verify

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

    seal = sub.add_parser("issue", help="seal the current (or a given) issue; optionally commit and publish it")
    seal.add_argument("--release", type=Path, required=True)
    seal.add_argument("--private", type=Path, required=True)
    seal.add_argument("--data", type=Path, required=True, help="engine data directory (candle cache, issues)")
    seal.add_argument("--issue", type=int, help="issue number (hours since the epoch); default: the current hour")
    seal.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    seal.add_argument("--wait-minutes", type=float, default=10.0, help="how long to wait for the candles")
    seal.add_argument("--commit", action="store_true", help="commit the root on-chain (key: LAPLACE_COMMIT_KEY_FILE)")
    seal.add_argument("--publish", action="store_true", help="publish committed issues (credentials: LAPLACE_R2_*)")
    seal.add_argument("--rpc", default=chain.RPC_URL)
    seal.add_argument("--ledger", default=chain.LEDGER)

    check = sub.add_parser("verify", help="check an issue's files, and that they were committed in time")
    check.add_argument("--issue", type=int, required=True)
    check.add_argument("--data", type=Path, help="read the files from this data directory instead of the public URL")
    check.add_argument("--url", default=publish.PUBLIC_URL)
    check.add_argument("--offline", action="store_true", help="check the files only, not the chain")
    check.add_argument("--rpc", default=chain.RPC_URL)
    check.add_argument("--ledger", default=chain.LEDGER)

    status = sub.add_parser("status", help="recent issues: sealed, committed, published")
    status.add_argument("--data", type=Path, required=True)
    status.add_argument("--hours", type=int, default=24)
    return parser


def cmd_release(args: argparse.Namespace) -> int:
    from laplace_engine.freeze import freeze

    frozen = freeze(args.run, args.name)
    release.write(frozen, args.out, args.private)
    for entry in frozen.describe():
        print(f"{entry['name']:12s} {entry['kind']:8s} {entry['id']}")
    return 0


def seal_issue(args: argparse.Namespace, number: int, target: Path) -> None:
    started = time.monotonic()
    rel = release.load(args.release, args.private)
    args.data.mkdir(parents=True, exist_ok=True)
    # A live issue waits for its candles until the deadline; a past issue (a dry run) only needs a moment.
    now = datetime.now(UTC)
    deadline = max(candles.issue_time(number) + timedelta(minutes=args.wait_minutes), now + timedelta(minutes=1))
    panel = candles.wait_for(number, args.data / "candles", deadline=deadline)
    with tempfile.TemporaryDirectory(dir=args.data) as workdir:
        finals, failures = issue.run_release(rel, panel, Path(workdir), workers=args.workers)
    ledger = issue.Ledger(chain.CHAIN_ID, args.ledger)
    sealed = issue.seal(number, rel, panel, finals, failures, ledger=ledger, commit=issue.engine_commit())
    issue.write_atomic(target / "forecasts.json", sealed.forecasts)
    issue.write_atomic(target / "manifest.json", sealed.manifest)
    for name, error in sealed.failures.items():
        log.warning("issue %d: %s failed: %s", number, name, error)
    root, manifest = sealed.root.hex(), sealed.manifest_hash.hex()
    seconds = time.monotonic() - started
    print(f"issue {number}: {sealed.leaves} forecasts, root 0x{root}, manifest 0x{manifest} ({seconds:.1f}s)")


def commit_issue(args: argparse.Namespace, number: int, target: Path) -> bool:
    closes = candles.issue_time(number) + timedelta(seconds=chain.COMMIT_WINDOW_S)
    if datetime.now(UTC) >= closes:
        log.error("issue %d: the commit window closed at %s; it stays uncommitted", number, f"{closes:%H:%M:%S}")
        return False
    manifest = (target / "manifest.json").read_bytes()
    body = json.loads(manifest)
    ledger = chain.Ledger(chain.RPC(args.rpc), args.ledger)
    account = chain.load_account(Path(os.environ["LAPLACE_COMMIT_KEY_FILE"]))
    commitment = ledger.commit(
        account, number, bytes.fromhex(body["root"][2:]), bytes(keccak(manifest)), int(body["leaves"])
    )
    issue.write_atomic(target / "commitment.json", commitment.to_json())
    print(f"issue {number}: committed in {commitment.tx} (block {commitment.block})")
    balance = ledger.balance(account.address)
    if balance < chain.LOW_BALANCE_WEI:
        log.warning("the commit key holds %.6f ETH; top it up", balance / 1e18)
    return True


def cmd_issue(args: argparse.Namespace) -> int:
    number = args.issue if args.issue is not None else candles.current_issue()
    target = issue_dir(args.data, number)
    if (target / "manifest.json").exists():
        log.info("issue %d is already sealed", number)  # never reseal: what may be committed is on disk
    else:
        seal_issue(args, number, target)
    ok = True
    if args.commit and not (target / "commitment.json").exists():
        try:
            ok = commit_issue(args, number, target)
        except Exception:
            log.exception("issue %d: commit failed", number)  # still publish the backlog below
            ok = False
    if args.publish:
        store = publish.R2.from_env()
        name = json.loads((args.release / release.RELEASE_FILE).read_text())["release"]
        publish.publish_release(store, args.release, name, args.data / "releases" / f"{name}.published")
        publish.publish_pending(store, args.data / "issues")
    return 0 if ok else 1


def cmd_verify(args: argparse.Namespace) -> int:
    if args.data is not None:
        target = issue_dir(args.data, args.issue)
        manifest, forecasts = (target / "manifest.json").read_bytes(), (target / "forecasts.json").read_bytes()
    else:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            fetched = [
                client.get(f"{args.url}/issues/{args.issue}/{name}") for name in ("manifest.json", "forecasts.json")
            ]
        for response in fetched:
            response.raise_for_status()
        manifest, forecasts = fetched[0].content, fetched[1].content
    result = verify.verify(manifest, forecasts)
    print(f"issue {result.issue}: {result.leaves} forecasts and {result.pools_checked} pool(s) check out")
    print(f"  root 0x{result.root.hex()}, manifest 0x{result.manifest_hash.hex()}")
    if args.offline:
        return 0
    found = chain.Ledger(chain.RPC(args.rpc), args.ledger).find(args.issue)
    if found is None:
        print(f"issue {args.issue}: no commitment on-chain")
        return 1
    checks = {
        "root": found.root == "0x" + result.root.hex(),
        "manifest hash": found.manifest == "0x" + result.manifest_hash.hex(),
        "leaf count": found.leaves == result.leaves,
        "commit window": found.in_window,
    }
    stamp = datetime.fromtimestamp(found.timestamp, UTC)
    for name, ok in checks.items():
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    print(f"committed in {found.tx}, block {found.block}, {stamp:%Y-%m-%d %H:%M:%S} UTC")
    return 0 if all(checks.values()) else 1


def cmd_status(args: argparse.Namespace) -> int:
    newest = candles.current_issue()
    for number in range(newest - args.hours + 1, newest + 1):
        target = issue_dir(args.data, number)
        flags = [
            name if (target / f).exists() else "-" * len(name)
            for name, f in (("sealed", "manifest.json"), ("committed", "commitment.json"), ("published", "published"))
        ]
        failed = ""
        if (target / "manifest.json").exists():
            failures = json.loads((target / "manifest.json").read_bytes())["failed"]
            failed = f"  failed: {', '.join(failures)}" if failures else ""
        print(f"{number} {candles.issue_time(number):%Y-%m-%d %H:%M}  {' '.join(flags)}{failed}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    commands = {"release": cmd_release, "issue": cmd_issue, "verify": cmd_verify, "status": cmd_status}
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
