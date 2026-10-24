"""Command line: freeze a release; seal, commit, publish and settle issues; verify an issue; scores; recent issues."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from eth_hash.auto import keccak

from laplace_engine import candles, chain, files, issue, publish, release, settle, verify
from laplace_research.data.candles import fetch_hourly
from laplace_research.evolve.feed import product

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
    seal.add_argument("--settle", action="store_true", help="settle and score every question that has become due")
    seal.add_argument("--rpc", default=chain.RPC_URL)
    seal.add_argument("--ledger", default=chain.LEDGER)

    check = sub.add_parser("verify", help="check an issue's files, and that they were committed in time")
    check.add_argument("--issue", type=int, required=True)
    check.add_argument("--data", type=Path, help="read the files from this data directory instead of the public URL")
    check.add_argument("--url", default=publish.PUBLIC_URL)
    check.add_argument("--offline", action="store_true", help="check the files only, not the chain")
    check.add_argument("--rpc", default=chain.RPC_URL)
    check.add_argument("--ledger", default=chain.LEDGER)

    board = sub.add_parser("scoreboard", help="mean log scores and gains over the baselines, from settled questions")
    board.add_argument("--data", type=Path, required=True)

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
    files.write_atomic(target / "forecasts.json", sealed.forecasts)
    files.write_atomic(target / "manifest.json", sealed.manifest)
    for name, error in sealed.failures.items():
        log.warning("issue %d: %s failed: %s", number, name, error)
    root, manifest = sealed.root.hex(), sealed.manifest_hash.hex()
    seconds = time.monotonic() - started
    print(f"issue {number}: {sealed.leaves} forecasts, root 0x{root}, manifest 0x{manifest} ({seconds:.1f}s)")


JOURNAL = "commit-tx.json"  # the signed commit transaction's hash, written before it is sent
GAP = "gap.json"  # an issue whose window closed without a commitment
CONFLICT = "conflict.json"  # an issue committed with a root other than its local files (never published)


def commit_issue(ledger: chain.Ledger, number: int, target: Path) -> bool:
    """Commit the issue, retrying transient failures until shortly before its window closes."""
    closes = candles.issue_time(number) + timedelta(seconds=chain.COMMIT_WINDOW_S)
    if (closes - datetime.now(UTC)).total_seconds() <= 30:
        log.error("issue %d: the commit window closed at %s; it stays uncommitted", number, f"{closes:%H:%M:%S}")
        return False
    key_file = os.environ.get("LAPLACE_COMMIT_KEY_FILE")
    if not key_file:
        raise RuntimeError("LAPLACE_COMMIT_KEY_FILE is not set")
    account = chain.load_account(Path(key_file))
    manifest = (target / "manifest.json").read_bytes()
    body = json.loads(manifest)
    while True:
        remaining = (closes - datetime.now(UTC)).total_seconds()
        if remaining <= 30:
            log.error("issue %d: the commit window closes at %s; it stays uncommitted", number, f"{closes:%H:%M:%S}")
            return False
        try:
            commitment = ledger.commit(
                account,
                number,
                bytes.fromhex(body["root"][2:]),
                bytes(keccak(manifest)),
                int(body["leaves"]),
                journal=target / JOURNAL,
                timeout_s=min(120.0, remaining - 20),
            )
        except chain.LedgerRevert as exc:
            # The chain's clock can trail ours by a few seconds at the start of the hour; any other refusal is final.
            if exc.reason == "OutsideCommitWindow" and remaining > chain.COMMIT_WINDOW_S - 120:
                time.sleep(5.0)
                continue
            log.error("issue %d: %s", number, exc)
            return False
        except Exception:
            log.exception("issue %d: commit attempt failed; retrying", number)
            time.sleep(min(15.0, max(remaining - 30, 1.0)))
            continue
        files.write_atomic(target / "commitment.json", commitment.to_json())
        print(f"issue {number}: committed in {commitment.tx} (block {commitment.block})")
        try:
            balance = ledger.balance(account.address)
            if balance < chain.LOW_BALANCE_WEI:
                log.warning("the commit key holds %.6f ETH; top it up", balance / 1e18)
        except Exception:
            log.exception("could not read the commit key's balance")
        return True


def reconcile(ledger: chain.Ledger, issues_dir: Path, current: int) -> None:
    """Resolve every sealed issue without a recorded commitment once its window has closed: a commitment that reached
    the chain without being recorded (a lost reply, a crash) is recovered; one that never did is marked as a gap."""
    if not issues_dir.exists():
        return
    for issue_dir in sorted((d for d in issues_dir.iterdir() if d.name.isdigit()), key=lambda d: int(d.name)):
        number = int(issue_dir.name)
        resolved = any((issue_dir / m).exists() for m in ("commitment.json", GAP, CONFLICT))
        if number >= current or resolved or not (issue_dir / "manifest.json").exists():
            continue
        try:
            manifest = (issue_dir / "manifest.json").read_bytes()
            body = json.loads(manifest)
            root, manifest_hash = bytes.fromhex(body["root"][2:]), bytes(keccak(manifest))
            onchain = ledger.root_of(number)
            if onchain == root:
                found = ledger.recover(number, root, manifest_hash, int(body["leaves"]), journal=issue_dir / JOURNAL)
                files.write_atomic(issue_dir / "commitment.json", found.to_json())
                log.warning("issue %d: recovered a commitment that had not been recorded (%s)", number, found.tx)
            elif onchain == bytes(32):
                files.write_atomic(issue_dir / GAP, release.canonical({"issue": number, "reason": "not committed"}))
                log.warning("issue %d: never committed; marked as a gap", number)
            else:
                record = {"issue": number, "local_root": body["root"], "onchain_root": "0x" + onchain.hex()}
                files.write_atomic(issue_dir / CONFLICT, release.canonical(record))
                log.error("issue %d: the committed root differs from the local files; not publishing it", number)
        except Exception:
            log.exception("issue %d: could not reconcile; will retry", number)


@contextmanager
def run_lock(path: Path) -> Iterator[bool]:
    """An exclusive, non-blocking lock: two runs must never seal or commit the same hour."""
    with path.open("a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def cmd_issue(args: argparse.Namespace) -> int:
    args.data.mkdir(parents=True, exist_ok=True)
    with run_lock(args.data / ".lock") as locked:
        if not locked:
            log.warning("another run holds the lock; leaving this hour to it")
            return 0
        return run_issue(args)


def run_issue(args: argparse.Namespace) -> int:
    """Seal, commit, reconcile, publish, settle. Each step is isolated, so one failure never blocks the backlog."""
    number = args.issue if args.issue is not None else candles.current_issue()
    target = issue_dir(args.data, number)
    issues = args.data / "issues"
    ok = True
    if (target / "manifest.json").exists():
        log.info("issue %d is already sealed", number)  # never reseal: what may be committed is on disk
    else:
        try:
            seal_issue(args, number, target)
        except Exception:
            log.exception("issue %d: sealing failed", number)
            ok = False
    if args.commit:
        ledger = chain.Ledger(chain.RPC(args.rpc), args.ledger)
        if (target / "manifest.json").exists() and not (target / "commitment.json").exists():
            try:
                ok = commit_issue(ledger, number, target) and ok
            except Exception:
                log.exception("issue %d: commit failed", number)
                ok = False
        reconcile(ledger, issues, candles.current_issue())
    store = None
    if args.publish:
        try:
            store = publish.R2.from_env()
            name = json.loads((args.release / release.RELEASE_FILE).read_text())["release"]
            publish.publish_release(store, args.release, name, args.data / "releases" / f"{name}.published")
            publish.publish_pending(store, issues)
        except Exception:
            log.exception("publishing failed; will retry next run")
            ok = False
    if args.settle and issues.exists():
        settled = settle.settle_due(issues, candles.current_issue(), coinbase_close(args.data / "candles"))
        if settled:
            log.info("settled %d question sets", len(settled))
        if store is not None:
            try:
                if publish.publish_settlements(store, issues):
                    publish.publish_scoreboard(store, settle.scoreboard(settle.load_settlements(issues)))
            except Exception:
                log.exception("publishing settlements failed; will retry next run")
                ok = False
    return 0 if ok else 1


def coinbase_close(cache_dir: Path) -> settle.FetchClose:
    """The close of the last public exchange candle opening at or before an hour (an hour without trades repeats the
    previous close), for questions whose settling issue was never committed."""
    client = httpx.Client(timeout=30.0, headers={"User-Agent": "laplace-engine"})

    def fetch(asset: str, open_time: int) -> float:
        hour = datetime.fromtimestamp(open_time, UTC)
        window = fetch_hourly(
            product(asset),
            hour - timedelta(hours=12),
            hour + timedelta(hours=1),
            client=client,
            cache_dir=cache_dir,
            max_filled_hours=12,
        )
        found = [c for c in window if c.ts <= open_time]
        if not found:
            raise ValueError(f"{asset}: no candle in the 12 hours up to {hour:%Y-%m-%d %H:%M}")
        return found[-1].close

    return fetch


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


def cmd_scoreboard(args: argparse.Namespace) -> int:
    board = settle.scoreboard(settle.load_settlements(args.data / "issues"))
    print(f"{board['settled_issues']} settled issues ({board['first_issue']} to {board['last_issue']})")
    print(
        f"{'forecaster':12s} {'questions':>9s} {'log score':>9s}  " + "  ".join(f"vs {b:>6s}" for b in settle.BASELINES)
    )
    for name, entry in sorted(board["forecasters"].items(), key=lambda kv: -kv[1]["mean_log_score"]):
        gains = "  ".join(f"{entry['vs'][b]['mean']:+9.4f}" if b in entry["vs"] else " " * 9 for b in settle.BASELINES)
        print(f"{name:12s} {entry['questions']:9d} {entry['mean_log_score']:9.4f}  {gains}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    newest = candles.current_issue()
    for number in range(newest - args.hours + 1, newest + 1):
        target = issue_dir(args.data, number)
        flags = [
            name if (target / f).exists() else "-" * len(name)
            for name, f in (("sealed", "manifest.json"), ("committed", "commitment.json"), ("published", "published"))
        ]
        flags += [name for name, f in (("GAP", GAP), ("CONFLICT", CONFLICT)) if (target / f).exists()]
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
    commands = {
        "release": cmd_release,
        "issue": cmd_issue,
        "verify": cmd_verify,
        "scoreboard": cmd_scoreboard,
        "status": cmd_status,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
