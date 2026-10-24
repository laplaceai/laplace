# engine

The live side of Laplace. Every hour it runs the current release's forecasters on the candles that have just
closed, rounds each forecast to whole basis points, and commits the Merkle root of the hour's forecasts to
`LaplaceLedger` on Robinhood Chain before any of them can settle. The forecasts and a manifest are then published, so
anyone can rebuild the root and check it against the chain. The design is in
[`docs/design/live-record.md`](../docs/design/live-record.md).

```bash
uv sync
uv run pytest

# freeze a research run into a release (public part here, private champions elsewhere)
uv run laplace-engine release --run ../research/runs/evolve-poc2 --name r1 --out releases/r1 --private PRIVATE_DIR
# seal the current hour's issue into DATA_DIR/issues/<issue>/ (or a past one with --issue)
uv run laplace-engine issue --release releases/r1 --private PRIVATE_DIR --data DATA_DIR
# check a published issue: rebuild it from its files, then check its commitment on-chain
uv run laplace-engine verify --issue ISSUE
```

## Operations
The hourly run is a systemd user timer (`deploy/laplace-issue.timer`, 30 seconds after every hour) that runs
`laplace-engine issue --commit --publish` from a dedicated worktree, so development never changes what runs.
`deploy/deploy.sh [ref]` checks out a ref there (default `origin/main`), syncs the locked dependencies and installs the
units. The service reads its secrets from an environment file outside the repository:

| variable | |
|---|---|
| `LAPLACE_COMMIT_KEY_FILE` | JSON file with the commit key (`privateKey`); the key only pays gas for `commit` |
| `LAPLACE_R2_ENDPOINT`, `LAPLACE_R2_BUCKET` | the public bucket's S3 endpoint and name |
| `LAPLACE_R2_ACCESS_KEY_ID`, `LAPLACE_R2_SECRET_ACCESS_KEY` | write access to that bucket |

```bash
systemctl --user list-timers laplace-issue.timer    # next run
journalctl --user -u laplace-issue -n 50            # last runs
uv run laplace-engine status --data DATA_DIR        # sealed, committed, published, per hour
```

Each run seals the issue (never resealing one that exists on disk), commits it if the window is still open (a rerun
recovers an existing commitment instead of sending a second transaction), then publishes every committed issue that
is not yet public. A missed hour stays a visible gap in the ledger.

| module | |
|---|---|
| `record.py` | leaves and the Merkle tree, mirrored by `LaplaceLedger.leafHash` and `verifyForecast` |
| `quantize.py` | rounding to basis points; exact integer pooling of rounded forecasts |
| `release.py`, `freeze.py` | releases: forecasters identified by their source or pool specification; freezing a research run |
| `baselines.py` | the baseline ladder as programs with frozen parameters |
| `candles.py`, `issue.py` | an issue's candles, its forecasts, root and manifest |
| `verify.py` | the reference verifier |
| `chain.py`, `publish.py` | committing to the ledger; publishing to `data.laplaceprotocol.com` |
| `vectors.py` | test vectors shared with the contracts (`python -m laplace_engine.vectors`) |

The research package (`../research`) supplies the feed definition, the sandboxed evaluator and the baselines; this
package adds only what the live record needs.
