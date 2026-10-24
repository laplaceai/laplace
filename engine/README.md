# engine

The live side of Laplace. Every hour it runs the current release's forecasters on the candles that have just
closed, rounds each forecast to whole basis points, and commits the Merkle root of the hour's forecasts to
`LaplaceLedger` on Robinhood Chain before any of them can settle. The forecasts and a manifest are then published, so
anyone can rebuild the root and check it against the chain.

The record format (leaves, tree, proofs) is defined in [`record.py`](src/laplace_engine/record.py) and mirrored by
`LaplaceLedger.leafHash` and `verifyForecast`. Shared test vectors keep the two in step:
`python -m laplace_engine.vectors` writes `contracts/test/vectors/record.json`, which Foundry checks against the
contract and pytest checks against this code.

```bash
uv sync
uv run pytest
```

The research package (`../research`) supplies the feed definition, the sandboxed evaluator and the baselines; this
package adds only what the live record needs.
