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
# rebuild an issue's leaves, root and pools from its files
uv run laplace-engine verify --data DATA_DIR --issue 498019
```

| module | |
|---|---|
| `record.py` | leaves and the Merkle tree, mirrored by `LaplaceLedger.leafHash` and `verifyForecast` |
| `quantize.py` | rounding to basis points; exact integer pooling of rounded forecasts |
| `release.py`, `freeze.py` | releases: forecasters identified by their source or pool specification; freezing a research run |
| `baselines.py` | the baseline ladder as programs with frozen parameters |
| `candles.py`, `issue.py` | an issue's candles, its forecasts, root and manifest |
| `verify.py` | the reference verifier |
| `vectors.py` | test vectors shared with the contracts (`python -m laplace_engine.vectors`) |

The research package (`../research`) supplies the feed definition, the sandboxed evaluator and the baselines; this
package adds only what the live record needs.
