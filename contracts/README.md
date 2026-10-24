# contracts

Solidity contracts for Laplace, built and tested with [Foundry](https://getfoundry.sh).

## LaplaceLedger
An append-only record of hourly forecasts. At the start of every hour the engine commits the Merkle root over all
forecasts made at that hour, plus the hash of a manifest describing how to verify them. The contract only accepts a
commitment from its committer, only for an issue later than the last one, and only within the first 15 minutes of
the issue hour, so the block timestamp proves each root was sealed before the shortest (one-hour) outcome could be
known. Missed hours stay visible as gaps.

Roots stay on-chain (`rootOf`), so other contracts can use the forecasts: `verifyForecast` checks one forecast and
its Merkle proof against the committed root, and `leafHash` defines the leaf encoding. Leaves and proofs follow
OpenZeppelin's `MerkleProof` conventions. The engine (`../engine`) builds the same trees in Python;
`test/vectors/record.json` is generated there and checked here, so the two implementations cannot drift apart.

```bash
forge build
forge test
```

Deploy (the committer address is the constructor argument; keys are never stored in this repository):
```bash
forge create src/LaplaceLedger.sol:LaplaceLedger --rpc-url "$RPC_URL" --private-key "$DEPLOYER_KEY" \
  --broadcast --constructor-args "$COMMITTER"
```
